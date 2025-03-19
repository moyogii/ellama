#[cfg(feature = "tts")]
use crate::sessions::SharedTts;

use crate::{
    easymark::MemoizedEasymarkHighlighter,
    widgets::{self, ModelPicker},
};
use anyhow::{Context, Result};
use eframe::egui::{
    self, pos2, vec2, Align, Color32, Frame, Key, KeyboardShortcut, Layout, Margin, Modifiers,
    Pos2, Rect, CornerRadius, Stroke, TextStyle, StrokeKind
};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use egui_modal::{Icon, Modal};
use egui_virtual_list::VirtualList;
use flowync::{error::Compact, CompactFlower, CompactHandle};
use ollama_rs::{
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, ChatMessageResponseStream},
        images::Image,
        options::GenerationOptions,
    },
    Ollama,
};
use std::{
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio_stream::StreamExt;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum Role {
    User,
    Assistant,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Message {
    model_name: String,
    content: String,
    role: Role,
    #[serde(skip)]
    is_generating: bool,
    #[serde(skip)]
    requested_at: Instant,
    time: chrono::DateTime<chrono::Utc>,
    #[serde(skip)]
    clicked_copy: bool,
    is_error: bool,
    is_reasoning: bool,
    #[serde(skip)]
    is_speaking: bool,
    images: Vec<PathBuf>,
    is_prepending: bool,
    #[serde(default)]
    reasoning_content: String,
    #[serde(skip)]
    start_reason_time: Instant,
    #[serde(default)]
    end_reason_time: f64,
}

impl Default for Message {
    fn default() -> Self {
        Self {
            content: String::new(),
            role: Role::User,
            is_generating: false,
            requested_at: Instant::now(),
            time: chrono::Utc::now(),
            clicked_copy: false,
            is_error: false,
            is_speaking: false,
            model_name: String::new(),
            images: Vec::new(),
            is_reasoning: false,
            is_prepending: false,
            reasoning_content: String::new(),
            start_reason_time: Instant::now(),
            end_reason_time: 0.0
        }
    }
}

#[cfg(feature = "tts")]
fn tts_control(tts: SharedTts, text: String, speak: bool) {
    std::thread::spawn(move || {
        if let Some(tts) = tts {
            if speak {
                let _ = tts
                    .write()
                    .speak(text, true)
                    .map_err(|e| log::error!("failed to speak: {e}"));
            } else {
                let _ = tts
                    .write()
                    .stop()
                    .map_err(|e| log::error!("failed to stop tts: {e}"));
            }
        }
    });
}

/// Convert a model name into a short name.
///
/// # Example
///
/// - nous-hermes2:latest -> Nous
/// - gemma:latest -> Gemma
/// - starling-lm:7b-beta-q5_K_M -> Starling
/// - bambucha/saiga-llama3 -> Saiga
fn make_short_name(name: &str) -> String {
    let mut c = name
        .split('/')
        .nth(1)
        .unwrap_or(name)
        .chars()
        .take_while(|c| c.is_alphanumeric());
    match c.next() {
        None => "Llama".to_string(),
        Some(f) => f.to_uppercase().collect::<String>() + c.collect::<String>().as_str(),
    }
}

enum MessageAction {
    None,
    Retry(usize),
    Regenerate(usize),
}

impl Message {
    #[inline]
    fn user(content: String, model_name: String, images: Vec<PathBuf>) -> Self {
        Self {
            content,
            role: Role::User,
            is_generating: false,
            model_name,
            images,
            ..Default::default()
        }
    }

    #[inline]
    fn assistant(content: String, model_name: String) -> Self {
        Self {
            content,
            role: Role::Assistant,
            is_generating: true,
            model_name,
            ..Default::default()
        }
    }

    #[inline]
    const fn is_user(&self) -> bool {
        matches!(self.role, Role::User)
    }

    fn show(
        &mut self,
        ui: &mut egui::Ui,
        commonmark_cache: &mut CommonMarkCache,
        #[cfg(feature = "tts")] tts: SharedTts,
        idx: usize,
        prepend_buf: &mut String,
    ) -> MessageAction {
        // message role
        let message_offset = ui
            .horizontal(|ui| {
                if self.is_user() {
                    let f = ui.label("👤").rect.left();
                    ui.label("You").rect.left() - f
                } else {
                    let f = ui.label("🐱").rect.left();
                    let offset = ui
                        .label(make_short_name(&self.model_name))
                        .on_hover_text(&self.model_name)
                        .rect
                        .left()
                        - f;
                    ui.add_enabled(false, egui::Label::new(&self.model_name));
                    offset
                }
            })
            .inner;

        // for some reason commonmark creates empty space above it when created,
        // compensate for that
        let is_commonmark = !self.content.is_empty() && !self.is_error && !self.is_prepending && !self.is_reasoning;
        if is_commonmark {
            ui.add_space(-TextStyle::Body.resolve(ui.style()).size + 4.0);
        }

        // Reasoning 
        if !self.reasoning_content.is_empty() {
            ui.add_space(4.0);
            
            let total_time = if self.is_reasoning { self.start_reason_time.elapsed().as_secs_f64() } else { self.end_reason_time };

            let text = if self.is_reasoning {
               "💭 Thinking...".to_string()
            } else {
                format!("💭 Thought for {:.1} seconds", total_time)
            };

            ui.collapsing(text, |ui| {
                ui.vertical(|ui| {
                    egui::ScrollArea::vertical()
                        .max_height(200.0)
                        .show(ui, |ui| {
                            CommonMarkViewer::new()
                                .max_image_width(Some(450))
                                .show(ui, commonmark_cache, &self.reasoning_content);
                        });
                });
            });
        }

        // message content / spinner
        let mut action = MessageAction::None;
        ui.horizontal(|ui| {
            ui.add_space(message_offset);
            if self.content.is_empty() && self.is_generating && !self.is_error {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());

                    // show time spent waiting for response
                    ui.add_enabled(
                        false,
                        egui::Label::new(format!(
                            "{:.1}s",
                            self.requested_at.elapsed().as_secs_f64()
                        )),
                    )
                });
            } else if self.is_error {
                ui.label("An error occurred while requesting completion");
                if ui
                    .button("Retry")
                    .on_hover_text(
                        "Try to generate a response again. Make sure you have Ollama running",
                    )
                    .clicked()
                {
                    action = MessageAction::Retry(idx);
                }
            } else if self.is_prepending {
                let textedit = ui.add(
                    egui::TextEdit::multiline(prepend_buf).hint_text("Prepend text to response…"),
                );
                macro_rules! cancel_prepend {
                    () => {
                        self.is_prepending = false;
                        prepend_buf.clear();
                    };
                }
                if textedit.lost_focus() && ui.input(|i| i.key_pressed(Key::Escape)) {
                    cancel_prepend!();
                }
                ui.vertical(|ui| {
                    if ui
                        .button("🔄 Regenerate")
                        .on_hover_text(
                            "Generate the response again, \
                            the LLM will start after any prepended text",
                        )
                        .clicked()
                    {
                        self.content = prepend_buf.clone();
                        self.is_prepending = false;
                        self.is_generating = true;
                        action = MessageAction::Regenerate(idx);
                    }
                    if !prepend_buf.is_empty()
                        && ui
                            .button("\u{270f} Edit")
                            .on_hover_text(
                                "Edit the message in the context, but don't regenerate it",
                            )
                            .clicked()
                    {
                        self.content = prepend_buf.clone();
                        cancel_prepend!();
                    }
                    if ui.button("❌ Cancel").clicked() {
                        cancel_prepend!();
                    }
                });
            } else {
                CommonMarkViewer::new().max_image_width(Some(512)).show(
                    ui,
                    commonmark_cache,
                    &self.content,
                );
            }
        });

        // images
        if !self.images.is_empty() {
            if is_commonmark {
                ui.add_space(4.0);
            }
            ui.horizontal(|ui| {
                ui.add_space(message_offset);
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        crate::image::show_images(ui, &mut self.images, false);
                    });
                })
            });
            ui.add_space(8.0);
        }

        if self.is_prepending {
            return action;
        }

        // copy buttons and such
        let shift_held = !ui.ctx().wants_keyboard_input() && ui.input(|i| i.modifiers.shift);
        if !self.is_generating
            && !self.content.is_empty()
            && (!self.is_user() || shift_held)
            && !self.is_error
        {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(message_offset);
                let copy = ui
                    .add(
                        egui::Button::new(if self.clicked_copy { "✔" } else { "🗐" })
                            .small()
                            .fill(egui::Color32::TRANSPARENT),
                    )
                    .on_hover_text(if self.clicked_copy {
                        "Copied!"
                    } else {
                        "Copy message"
                    });
                if copy.clicked() {
                    ui.ctx().copy_text(self.content.clone());
                    self.clicked_copy = true;
                }
                self.clicked_copy = self.clicked_copy && copy.hovered();

                #[cfg(feature = "tts")]
                {
                    let speak = ui
                        .add(
                            egui::Button::new(if self.is_speaking { "…" } else { "🔊" })
                                .small()
                                .fill(egui::Color32::TRANSPARENT),
                        )
                        .on_hover_text("Read the message out loud. Right click to repeat");

                    if speak.clicked() {
                        if self.is_speaking {
                            self.is_speaking = false;
                            tts_control(tts, String::new(), false);
                        } else {
                            self.is_speaking = true;
                            tts_control(tts, self.content.clone(), true);
                        }
                    } else if speak.secondary_clicked() {
                        self.is_speaking = true;
                        tts_control(tts, self.content.clone(), true);
                    }
                }

                if !self.is_user()
                    && prepend_buf.is_empty()
                    && ui
                        .add(
                            egui::Button::new("🔄")
                                .small()
                                .fill(egui::Color32::TRANSPARENT),
                        )
                        .on_hover_text("Regenerate")
                        .clicked()
                {
                    prepend_buf.clear();
                    self.is_prepending = true;
                }
            });
        }
        ui.add_space(12.0);

        action
    }

    pub fn process_reasoning_content(&mut self) {
        if self.reasoning_content.is_empty() {
            self.start_reason_time = Instant::now();

            let mut processed_content = String::new();
            let mut current_reasoning = String::new();
            let mut is_reasoning = false;
            let mut content = self.content.clone();
            
            // Process all <think> tags
            while !content.is_empty() {
                if is_reasoning {
                    if let Some(end_pos) = content.find("</think>") {
                        let reasoning_text = &content[..end_pos];
                        current_reasoning.push_str(reasoning_text);
                        content = content[end_pos + 8..].to_string();
                        
                        self.end_reason_time = self.start_reason_time.elapsed().as_secs_f64();
                        is_reasoning = false;
                    } else {
                        current_reasoning.push_str(&content);
                        content.clear();
                    }
                } else {
                    if let Some(start_pos) = content.find("<think>") {
                        let before_reasoning = &content[..start_pos];
                        processed_content.push_str(before_reasoning);
                        content = content[start_pos + 7..].to_string();
                        is_reasoning = true;
                    } else {
                        processed_content.push_str(&content);
                        content.clear();
                    }
                }
            }
            
            if !current_reasoning.is_empty() {
                self.reasoning_content = current_reasoning.trim().to_string();
                self.content = processed_content;
            }
        }
    }
}

// <completion progress, final completion, error>
type CompletionFlower = CompactFlower<(usize, String), (usize, String), (usize, String)>;
type CompletionFlowerHandle = CompactHandle<(usize, String), (usize, String), (usize, String)>;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Chat {
    chatbox: String,
    #[serde(skip)]
    chatbox_height: f32,
    pub messages: Vec<Message>,
    #[serde(skip)]
    flower: CompletionFlower,
    #[serde(skip)]
    retry_message_idx: Option<usize>,
    pub summary: String,
    #[serde(skip)]
    chatbox_highlighter: MemoizedEasymarkHighlighter,
    stop_generating: Arc<AtomicBool>,
    #[serde(skip)]
    virtual_list: VirtualList,
    pub model_picker: ModelPicker,
    pub images: Vec<PathBuf>,
    prepend_buf: String,
}

impl Default for Chat {
    fn default() -> Self {
        Self {
            chatbox: String::new(),
            chatbox_height: 0.0,
            messages: Vec::new(),
            flower: CompletionFlower::new(1),
            retry_message_idx: None,
            summary: String::new(),
            chatbox_highlighter: MemoizedEasymarkHighlighter::default(),
            stop_generating: Arc::new(AtomicBool::new(false)),
            virtual_list:{
                let mut list = VirtualList::new();
                list.check_for_resize(false);
                list
            },
            model_picker: ModelPicker::default(),
            images: Vec::new(),
            prepend_buf: String::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn request_completion(
    ollama: Ollama,
    messages: Vec<ChatMessage>,
    handle: &CompletionFlowerHandle,
    stop_generating: Arc<AtomicBool>,
    selected_model: String,
    options: GenerationOptions,
    template: Option<String>,
    index: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "requesting completion... (history length: {})",
        messages.len()
    );

    // if any assistant message was prepended, save it so we can prepend it
    // to the final response
    let prepend = {
        if let Some(last) = messages.last() {
            if last.role == ollama_rs::generation::chat::MessageRole::Assistant {
                last.content.clone()
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    };

    let mut request = ChatMessageRequest::new(selected_model, messages).options(options);
    if let Some(template) = template {
        request = request.template(template);
    }
    let mut stream: ChatMessageResponseStream = ollama.send_chat_messages_stream(request).await?;

    log::info!("reading response...");

    let mut response = String::new();
    let mut is_whitespace = true;
    let mut is_reasoning = false;
    let mut reasoning_response = String::new();
    let mut current_reasoning_chunk = String::new();
    let mut thinking_tag_depth = 0;

    while let Some(Ok(res)) = stream.next().await {
        let msg = res.message;
        if is_whitespace && msg.content.trim().is_empty() {
            continue;
        }
        let content = if is_whitespace {
            msg.content.trim_start()
        } else {
            &msg.content
        };
        is_whitespace = false;

        // Process content chunk by chunk, looking for <think> tags
        let mut text_remaining = content.to_string();
        
        while !text_remaining.is_empty() {
            if is_reasoning {
                if let Some(end_pos) = text_remaining.find("</think>") {
                    let reasoning_content = &text_remaining[..end_pos];
                    current_reasoning_chunk.push_str(reasoning_content);
                    
                    handle.send((index, "\0REASONING_CONTENT:".to_string() + reasoning_content));
                    
                    text_remaining = text_remaining[end_pos + 8..].to_string();
                    
                    // Decrease thinking tag depth
                    thinking_tag_depth -= 1;
                    if thinking_tag_depth == 0 {
                        is_reasoning = false;
                        handle.send((index, "\0ENDREASONING".to_string()));
                        
                        reasoning_response.push_str(&current_reasoning_chunk);
                        reasoning_response.push_str("\n\n");
                        current_reasoning_chunk.clear();
                    }
                } else {
                    current_reasoning_chunk.push_str(&text_remaining);
                    handle.send((index, "\0REASONING_CONTENT:".to_string() + &text_remaining));
                    text_remaining.clear();
                }
            } else {
                if let Some(start_pos) = text_remaining.find("<think>") {
                    let before_reasoning = &text_remaining[..start_pos];
                    if !before_reasoning.is_empty() {
                        response.push_str(before_reasoning);
                        handle.send((index, before_reasoning.to_string()));
                    }
                    
                    is_reasoning = true;
                    thinking_tag_depth += 1;
                    handle.send((index, "\0REASONING".to_string()));
            
                    current_reasoning_chunk.clear();
                    text_remaining = text_remaining[start_pos + 7..].to_string();
                } else {
                    response.push_str(&text_remaining);
                    handle.send((index, text_remaining.clone()));
                    text_remaining.clear();
                }
            }
        }
        
        if stop_generating.load(Ordering::SeqCst) {
            log::info!("stopping generation");
            drop(stream);
            stop_generating.store(false, Ordering::SeqCst);
            break;
        }
    }

    if is_reasoning {
        handle.send((index, "\0ENDREASONING".to_string()));
    }

    log::info!(
        "completion request complete, response length: {}, reasoning response length: {}",
        response.len(),
        reasoning_response.len()
    );
    
    // Send the entire reasoning response
    if !reasoning_response.is_empty() {
        handle.send((index, "\0REASONINGRESPONSE:".to_string() + &reasoning_response.trim()));
    }
    
    handle.success((index, prepend + response.trim()));
    Ok(())
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub enum ChatExportFormat {
    #[default]
    Plaintext,
    Json,
    Ron,
}

impl std::fmt::Display for ChatExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ChatExportFormat {
    pub const ALL: [Self; 3] = [Self::Plaintext, Self::Json, Self::Ron];

    #[inline]
    pub const fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::Plaintext => &["txt"],
            Self::Json => &["json"],
            Self::Ron => &["ron"],
        }
    }
}

pub async fn export_messages(
    messages: Vec<Message>,
    format: ChatExportFormat,
    task: impl std::future::Future<Output = Option<rfd::FileHandle>>,
) -> Result<egui_notify::Toast> {
    let Some(file) = task.await else {
        log::info!("export cancelled");
        return Ok(egui_notify::Toast::info("Export cancelled"));
    };
    log::info!(
        "exporting {} messages to {file:?} (format: {format:?})...",
        messages.len()
    );

    let f = std::fs::File::create(file.path())?;
    let mut f = std::io::BufWriter::new(f);

    match format {
        ChatExportFormat::Plaintext => {
            for msg in &messages {
                writeln!(
                    f,
                    "{} - {:?} ({}): {} (Reasoning ({:.1} seconds)): {}",
                    msg.time.to_rfc3339(),
                    msg.role,
                    msg.model_name,
                    msg.content,
                    msg.end_reason_time,
                    msg.reasoning_content
                )?;

                writeln!(f)?; // Add blank line between messages
            }
        }
        ChatExportFormat::Json | ChatExportFormat::Ron => {
            // These formats automatically handle all fields through serialization
            match format {
                ChatExportFormat::Json => {
                    serde_json::to_writer_pretty(&mut f, &messages)?;
                }
                ChatExportFormat::Ron => {
                    ron::ser::to_writer_pretty(&mut f, &messages, ron::ser::PrettyConfig::default())?;
                }
                _ => unreachable!(),
            }
        }
    }

    f.flush().context("failed to flush writer")?;

    log::info!("export complete");
    Ok(egui_notify::Toast::success(format!(
        "Exported {} messages to {}",
        messages.len(),
        file.file_name(),
    )))
}

fn make_summary(prompt: &str) -> String {
    const MAX_SUMMARY_LENGTH: usize = 24;
    let mut summary = String::with_capacity(MAX_SUMMARY_LENGTH);
    for (i, ch) in prompt.chars().enumerate() {
        if i >= MAX_SUMMARY_LENGTH {
            summary.push('…');
            break;
        }
        if ch == '\n' {
            break;
        }
        if i == 0 {
            summary += &ch.to_uppercase().to_string();
        } else {
            summary.push(ch);
        }
    }
    summary
}

#[derive(Debug, Clone, Copy)]
pub enum ChatAction {
    None,
    PickImages { id: usize },
}

impl Chat {
    #[inline]
    pub fn new(id: usize, model_picker: ModelPicker) -> Self {
        Self {
            flower: CompletionFlower::new(id),
            model_picker,
            ..Default::default()
        }
    }

    pub fn process_all_reasoning_messages(&mut self) {
        for message in &mut self.messages {
            message.process_reasoning_content();
        }
    }
    
    #[inline]
    pub fn id(&self) -> usize {
        self.flower.id()
    }

    fn convert_images(images: &[PathBuf]) -> Option<Vec<Image>> {
        if !images.is_empty() {
            Some(
                images
                    .iter()
                    // TODO: handle errors
                    .map(|i| {
                        crate::image::convert_image(i)
                            .map_err(|e| log::error!("failed to convert image: {e}"))
                            .unwrap()
                    })
                    .collect(),
            )
        } else {
            None
        }
    }

    fn get_context_messages(messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|m| {
                let mut message = match m.role {
                    Role::User => ChatMessage::user(m.content.clone()),
                    Role::Assistant => ChatMessage::assistant(m.content.clone()),
                };

                // TODO: don't do this each time!
                message.images = Self::convert_images(&m.images);

                message
            })
            .collect()
    }

    fn send_message(&mut self, ollama: &Ollama) {
        // don't send empty messages
        if self.chatbox.is_empty() && self.images.is_empty() {
            return;
        }

        // remove old error messages
        self.messages.retain(|m| !m.is_error);

        let prompt = self.chatbox.trim_end().to_string();
        let model_name = self.model_picker.selected_model().to_owned();
        self.messages.push(Message::user(
            prompt.clone(),
            model_name.clone(),
            self.images.clone(),
        ));

        if self.summary.is_empty() {
            self.summary = make_summary(&prompt);
        }

        // clear chatbox & images
        self.chatbox.clear();
        self.images.clear();

        // get ready for assistant response
        self.messages
            .push(Message::assistant(String::new(), model_name.clone()));

        self.spawn_completion(
            ollama.clone(),
            Self::get_context_messages(&self.messages),
            model_name,
        );
    }

    /// spawn a new task to generate the completion
    fn spawn_completion(
        &self,
        ollama: Ollama,
        context_messages: Vec<ChatMessage>,
        model_name: String,
    ) {
        let handle = self.flower.handle(); // recv'd by gui thread
        let stop_generation = self.stop_generating.clone();
        let generation_options = self.model_picker.get_generation_options();
        let template = self.model_picker.template.clone();
        let index = self.messages.len() - 1;
        tokio::spawn(async move {
            handle.activate();
            let _ = request_completion(
                ollama,
                context_messages,
                &handle,
                stop_generation,
                model_name,
                generation_options,
                template,
                index,
            )
            .await
            .map_err(|e| {
                log::error!("failed to request completion: {e}");
                handle.error((index, e.to_string()));
            });
        });
    }

    fn regenerate_response(&mut self, ollama: &Ollama, idx: usize) {
        // remake context history to make the message we want to regenerate last
        let mut messages = Self::get_context_messages(&self.messages[..idx]);

        // start with the prepended message and update it in the displayed messages
        messages.push(ChatMessage::assistant(self.prepend_buf.clone()));
        self.messages[idx].content = self.prepend_buf.clone();
        self.prepend_buf.clear();

        // start completing the message
        self.spawn_completion(
            ollama.clone(),
            messages,
            self.messages[idx].model_name.clone(),
        );
    }

    fn show_chatbox(
        &mut self,
        ui: &mut egui::Ui,
        is_max_height: bool,
        is_generating: bool,
        ollama: &Ollama,
    ) -> ChatAction {
        let mut action = ChatAction::None;
        if let Some(idx) = self.retry_message_idx.take() {
            self.chatbox = self.messages[idx - 1].content.clone();
            self.messages.remove(idx); // remove assistant message
            self.messages.remove(idx - 1); // remove user message
            self.send_message(ollama);
        }

        if is_max_height {
            ui.add_space(8.0);
        }

        let images_height = if !self.images.is_empty() {
            ui.add_space(8.0);
            let height = ui
                .horizontal(|ui| {
                    crate::image::show_images(ui, &mut self.images, true);
                })
                .response
                .rect
                .height();
            height + 16.0
        } else {
            0.0
        };

        ui.horizontal_centered(|ui| {
            if ui
                .add(
                    egui::Button::new("➕")
                        .min_size(vec2(32.0, 32.0))
                        .corner_radius(CornerRadius::same(u8::MAX)),
                )
                .on_hover_text_at_pointer("Pick Images")
                .clicked()
            {
                action = ChatAction::PickImages { id: self.id() };
            }
            ui.with_layout(
                Layout::left_to_right(Align::Center).with_main_justify(true),
                |ui| {
                    let Self {
                        chatbox_highlighter: highlighter,
                        ..
                    } = self;
                    let mut layouter = |ui: &egui::Ui, easymark: &str, wrap_width: f32| {
                        let mut layout_job = highlighter.highlight(ui.style(), easymark);
                        layout_job.wrap.max_width = wrap_width;
                        ui.fonts(|f| f.layout_job(layout_job))
                    };

                    self.chatbox_height = egui::TextEdit::multiline(&mut self.chatbox)
                        .return_key(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter))
                        .hint_text("Ask me anything…")
                        .layouter(&mut layouter)
                        .show(ui)
                        .response
                        .rect
                        .height()
                        + images_height;
                    if !is_generating
                        && ui.input(|i| i.key_pressed(Key::Enter) && i.modifiers.is_none())
                    {
                        self.send_message(ollama);
                    }
                },
            );
        });

        if is_max_height {
            ui.add_space(8.0);
        }

        action
    }

    #[inline]
    pub fn flower_active(&self) -> bool {
        self.flower.is_active()
    }

    pub fn poll_flower(&mut self, modal: &mut Modal) {
        self.flower
            .extract(|(idx, progress)| {
                let message = &mut self.messages[idx];
                match progress.as_str() {
                    "\0REASONING" => {
                        message.is_reasoning = true;
                    }
                    "\0ENDREASONING" => {
                        message.end_reason_time = message.start_reason_time.elapsed().as_secs_f64();
                        message.is_reasoning = false;
                    }
                    content if content.starts_with("\0REASONINGRESPONSE:") => {
                        message.reasoning_content = content["\0REASONINGRESPONSE:".len()..].to_string();
                    }
                    content if content.starts_with("\0REASONING_CONTENT:") => {
                        let new_content = &content["\0REASONING_CONTENT:".len()..];
                        if !new_content.is_empty() {
                            message.reasoning_content.push_str(new_content);
                        }
                    }
                    _ => message.content += progress.as_str()
                }
            })
            .finalize(|result| {
                if let Ok((idx, content)) = result {
                    let message = &mut self.messages[idx];
                    message.content = content.clone();
                    message.is_generating = false;
                    message.is_reasoning = false;

                    if message.content.trim().is_empty() && !message.reasoning_content.is_empty() {
                        message.content = message.reasoning_content.clone();
                        message.reasoning_content.clear();
                    }
                    
                    if !message.reasoning_content.is_empty() {
                        message.reasoning_content = message.reasoning_content.trim().to_string();
                    }
                } else if let Err(e) = result {
                    let (idx, msg) = match e {
                        Compact::Panicked(e) => {
                            (self.messages.len() - 1, format!("Tokio task panicked: {e}"))
                        }
                        Compact::Suppose((idx, e)) => (idx, e),
                    };
                    let message = &mut self.messages[idx];
                    message.content = msg.clone();
                    message.is_error = true;
                    modal
                        .dialog()
                        .with_body(msg)
                        .with_title("Failed to generate completion!")
                        .with_icon(Icon::Error)
                        .open();
                    message.is_generating = false;
                    message.is_reasoning = false;
                }
            });
    }

    pub fn last_message_contents(&self) -> Option<String> {
        for message in self.messages.iter().rev() {
            if message.content.is_empty() {
                continue;
            }
            return Some(if message.is_user() {
                format!("You: {}", message.content)
            } else {
                message.content.to_string()
            });
        }
        None
    }

    fn stop_generating_button(&self, ui: &mut egui::Ui, radius: f32, pos: Pos2) {
        let rect = Rect::from_min_max(pos + vec2(-radius, -radius), pos + vec2(radius, radius));
        let (hovered, primary_clicked) = ui.input(|i| {
            (
                i.pointer
                    .interact_pos()
                    .map(|p| rect.contains(p))
                    .unwrap_or(false),
                i.pointer.primary_clicked(),
            )
        });
        if hovered && primary_clicked {
            self.stop_generating.store(true, Ordering::SeqCst);
        } else {
            ui.painter().circle(
                pos,
                radius,
                if hovered {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    if ui.style().visuals.dark_mode {
                        let c = ui.style().visuals.faint_bg_color;
                        Color32::from_rgb(c.r(), c.g(), c.b())
                    } else {
                        Color32::WHITE
                    }
                } else {
                    ui.style().visuals.window_fill
                },
                Stroke::new(2.0, ui.style().visuals.window_stroke.color),
            );
            ui.painter().rect_stroke(
                rect.shrink(radius / 2.0 + 1.2),
                2,
                Stroke::new(2.0, Color32::DARK_GRAY),
                StrokeKind::Inside,
            );
        }
    }

    fn show_chat_scrollarea(
        &mut self,
        ui: &mut egui::Ui,
        ollama: &Ollama,
        commonmark_cache: &mut CommonMarkCache,
        #[cfg(feature = "tts")] tts: SharedTts,
    ) -> Option<usize> {
        let mut new_speaker: Option<usize> = None;
        let mut any_prepending = false;
        let mut regenerate_response_idx = None;
        egui::ScrollArea::both()
            .stick_to_bottom(true)
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.add_space(16.0);
                self.virtual_list
                    .ui_custom_layout(ui, self.messages.len(), |ui, index| {
                        let Some(message) = self.messages.get_mut(index) else {
                            return 0;
                        };
                        let prev_speaking = message.is_speaking;
                        if any_prepending && message.is_prepending {
                            message.is_prepending = false;
                        }
                        let action = message.show(
                            ui,
                            commonmark_cache,
                            #[cfg(feature = "tts")]
                            tts.clone(),
                            index,
                            &mut self.prepend_buf,
                        );
                        match action {
                            MessageAction::None => (),
                            MessageAction::Retry(idx) => {
                                self.retry_message_idx = Some(idx);
                            }
                            MessageAction::Regenerate(idx) => {
                                regenerate_response_idx = Some(idx);
                            }
                        }
                        any_prepending |= message.is_prepending;
                        if !prev_speaking && message.is_speaking {
                            new_speaker = Some(index);
                        }
                        1 // 1 rendered item per row
                    });
            });
        if let Some(regenerate_idx) = regenerate_response_idx {
            self.regenerate_response(ollama, regenerate_idx);
        }
        new_speaker
    }

    fn send_text(&mut self, ollama: &Ollama, text: &str) {
        self.chatbox = text.to_owned();
        self.send_message(ollama);
    }

    fn show_suggestions(&mut self, ui: &mut egui::Ui, ollama: &Ollama) {
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            widgets::centerer(ui, |ui| {
                let avail_width = ui.available_rect_before_wrap().width() - 24.0;
                ui.horizontal(|ui| {
                    ui.heading("Ellama");
                    if !self.model_picker.selected_model().is_empty() {
                        ui.add_enabled_ui(false, |ui| {
                            ui.heading(format!("({})", self.model_picker.selected_model()));
                        });
                    }
                });
                egui::Grid::new("suggestions_grid")
                    .num_columns(3)
                    .max_col_width((avail_width / 2.0).min(200.0))
                    .spacing(vec2(6.0, 6.0))
                    .show(ui, |ui| {
                        if widgets::suggestion(ui, "Tell me a fun fact", "about the Roman empire")
                            .clicked()
                        {
                            self.send_text(ollama, "Tell me a fun fact about the Roman empire");
                        }
                        if widgets::suggestion(
                            ui,
                            "Show me a code snippet",
                            "of a web server in Rust",
                        )
                        .clicked()
                        {
                            self.send_text(
                                ollama,
                                "Show me a code snippet of a web server in Rust",
                            );
                        }
                        widgets::dummy(ui);
                        ui.end_row();

                        if widgets::suggestion(ui, "Tell me a joke", "about crabs").clicked() {
                            self.send_text(ollama, "Tell me a joke about crabs");
                        }
                        if widgets::suggestion(ui, "Give me ideas", "for a birthday present")
                            .clicked()
                        {
                            self.send_text(ollama, "Give me ideas for a birthday present");
                        }
                        widgets::dummy(ui);
                        ui.end_row();
                    });
            });
        });
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        ollama: &Ollama,
        #[cfg(feature = "tts")] tts: SharedTts,
        #[cfg(feature = "tts")] stopped_speaking: bool,
        commonmark_cache: &mut CommonMarkCache,
    ) -> ChatAction {
        self.process_all_reasoning_messages();
        
        let avail = ctx.available_rect();
        let max_height = avail.height() * 0.4 + 24.0;
        let chatbox_panel_height = self.chatbox_height + 24.0;
        let actual_chatbox_panel_height = chatbox_panel_height.min(max_height);
        let is_generating = self.flower_active();
        let mut action = ChatAction::None;

        egui::TopBottomPanel::bottom("chatbox_panel")
            .exact_height(actual_chatbox_panel_height)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    action = self.show_chatbox(
                        ui,
                        chatbox_panel_height >= max_height,
                        is_generating,
                        ollama,
                    );
                });
            });

        #[cfg(feature = "tts")]
        let mut new_speaker: Option<usize> = None;

        egui::CentralPanel::default()
            .frame(Frame::central_panel(&ctx.style()).inner_margin(Margin {
                left: 16,
                right: 16,
                top: 0,
                bottom: 3,
            }))
            .show(ctx, |ui| {
                if self.messages.is_empty() {
                    self.show_suggestions(ui, ollama);
                } else {
                    #[allow(unused_variables)]
                    if let Some(new) = self.show_chat_scrollarea(
                        ui,
                        ollama,
                        commonmark_cache,
                        #[cfg(feature = "tts")]
                        tts,
                    ) {
                        #[cfg(feature = "tts")]
                        {
                            new_speaker = Some(new);
                        }
                    }

                    // stop generating button
                    if is_generating {
                        self.stop_generating_button(
                            ui,
                            16.0,
                            pos2(
                                ui.cursor().max.x - 32.0,
                                avail.height() - 32.0 - actual_chatbox_panel_height,
                            ),
                        );
                    }
                }
            });

        #[cfg(feature = "tts")]
        {
            if let Some(new_idx) = new_speaker {
                log::debug!("new speaker {new_idx} appeared, updating message icons");
                for (i, msg) in self.messages.iter_mut().enumerate() {
                    if i == new_idx {
                        continue;
                    }
                    msg.is_speaking = false;
                }
            }
            if stopped_speaking {
                log::debug!("TTS stopped speaking, updating message icons");
                for msg in self.messages.iter_mut() {
                    msg.is_speaking = false;
                }
            }
        }

        action
    }
}
