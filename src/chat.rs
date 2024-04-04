use crate::{easymark::MemoizedEasymarkHighlighter, sessions::SharedTts};
use eframe::egui::{
    self, pos2, vec2, Align, Color32, Frame, Key, KeyboardShortcut, Layout, Margin, Modifiers,
    Pos2, Rect, Stroke,
};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use egui_modal::{Icon, Modal};
use flowync::{error::Compact, CompactFlower, CompactHandle};
use ollama_rs::{
    generation::chat::{request::ChatMessageRequest, ChatMessage, ChatMessageResponseStream},
    Ollama,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio_stream::StreamExt;

#[derive(Clone)]
struct Message {
    content: String,
    is_user: bool,
    is_generating: bool,
    requested_at: Instant,
    clicked_copy: bool,
    is_error: bool,
    is_speaking: bool,
}

impl Default for Message {
    fn default() -> Self {
        Self {
            content: String::new(),
            is_user: false,
            is_generating: false,
            requested_at: Instant::now(),
            clicked_copy: false,
            is_error: false,
            is_speaking: false,
        }
    }
}

const MESSAGE_ABORTED_TEXT: &str = "\\<aborted by user\\>";

fn tts_control(tts: SharedTts, mut text: String, speak: bool) {
    std::thread::spawn(move || {
        if let Some(tts) = tts {
            if speak {
                if text == MESSAGE_ABORTED_TEXT {
                    // don't speak backslashes
                    text = "Aborted by user".to_string();
                }
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

impl Message {
    #[inline]
    fn user(content: String) -> Self {
        Self {
            content,
            is_user: true,
            is_generating: false,
            ..Default::default()
        }
    }

    #[inline]
    fn assistant(content: String) -> Self {
        Self {
            content,
            is_user: false,
            is_generating: true,
            ..Default::default()
        }
    }

    fn show(
        &mut self,
        ui: &mut egui::Ui,
        commonmark_cache: &mut CommonMarkCache,
        tts: SharedTts,
        idx: usize,
    ) -> bool {
        // message role
        let message_offset = ui
            .horizontal(|ui| {
                if self.is_user {
                    let f = ui.label("👤").rect.left();
                    ui.label("You").rect.left() - f
                } else {
                    let f = ui.label("🐱").rect.left();
                    ui.label("Llama").rect.left() - f
                }
            })
            .inner;

        // for some reason commonmark creates empty space above it when created,
        // compensate for that
        if !self.content.is_empty() && !self.is_error {
            ui.add_space(-24.0);
        }

        // message content / spinner
        let mut retry = false;
        ui.horizontal(|ui| {
            ui.add_space(message_offset);
            if self.content.is_empty() && !self.is_error {
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
                retry = ui
                    .button("Retry")
                    .on_hover_text(
                        "Try to generate a response again. Make sure you have Ollama running",
                    )
                    .clicked();
            } else {
                CommonMarkViewer::new(format!("message_{idx}_commonmark"))
                    .max_image_width(Some(512))
                    .show(ui, commonmark_cache, &self.content);
            }
        });

        // copy buttons and such
        if !self.is_generating && !self.content.is_empty() && !self.is_user && !self.is_error {
            ui.add_space(-12.0);
            ui.horizontal(|ui| {
                ui.add_space(message_offset);
                let copy = ui
                    .add(
                        egui::Button::new(if self.clicked_copy { "✔" } else { "🗐" })
                            .small()
                            .fill(egui::Color32::TRANSPARENT),
                    )
                    .on_hover_text("Copy message");
                if copy.clicked() {
                    ui.ctx().copy_text(self.content.clone());
                    self.clicked_copy = true;
                }
                self.clicked_copy = self.clicked_copy && copy.hovered();

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
            });
            ui.add_space(8.0);
        }

        retry
    }
}

// <completion progress, final completion, error>
type CompletionFlower = CompactFlower<String, String, String>;
type CompletionFlowerHandle = CompactHandle<String, String, String>;

pub struct Chat {
    chatbox: String,
    chatbox_height: f32,
    messages: Vec<Message>,
    context_messages: Vec<ChatMessage>,
    flower: CompletionFlower,
    retry_message_idx: Option<usize>,
    pub summary: String,
    highlighter: MemoizedEasymarkHighlighter,
    stop_generating: Arc<AtomicBool>,
}

impl Default for Chat {
    fn default() -> Self {
        Self {
            chatbox: String::new(),
            chatbox_height: 0.0,
            messages: vec![],
            context_messages: vec![],
            flower: CompletionFlower::new(1),
            retry_message_idx: None,
            summary: String::new(),
            highlighter: MemoizedEasymarkHighlighter::default(),
            stop_generating: Arc::new(AtomicBool::new(false)),
        }
    }
}

async fn request_completion(
    ollama: Ollama,
    messages: Vec<ChatMessage>,
    handle: &CompletionFlowerHandle,
    stop_generating: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "requesting completion... (history length: {})",
        messages.len()
    );
    let mut stream: ChatMessageResponseStream = ollama
        .send_chat_messages_stream(ChatMessageRequest::new(
            "starling-lm:7b-alpha-q5_K_S".to_string(),
            messages,
        ))
        .await?;

    log::info!("reading response...");

    let mut response = String::new();
    let mut is_whitespace = true;

    while let Some(Ok(res)) = stream.next().await {
        if let Some(msg) = res.message {
            if is_whitespace && msg.content.trim().is_empty() {
                continue;
            }
            let content = if is_whitespace {
                msg.content.trim_start()
            } else {
                &msg.content
            };
            is_whitespace = false;

            // send message to gui thread
            handle.send(content.to_string());
            response += content;

            if stop_generating.load(Ordering::SeqCst) {
                log::info!("stopping generation");
                drop(stream);
                stop_generating.store(false, Ordering::SeqCst);
                break;
            }
        }
    }

    log::info!(
        "completion request complete, response length: {}",
        response.len()
    );
    if response.is_empty() {
        handle.success(MESSAGE_ABORTED_TEXT.to_string()); // prevent html tags
    } else {
        handle.success(response.trim().to_string());
    }
    Ok(())
}

impl Chat {
    fn send_message(&mut self, ollama: &Ollama) {
        // don't send empty messages
        if self.chatbox.is_empty() {
            return;
        }

        // remove old error messages
        self.messages.retain(|m| !m.is_error);

        let prompt = self.chatbox.trim_end().to_string();
        self.messages.push(Message::user(prompt.clone()));

        if self.summary.len() < 24 {
            for word in prompt.split_whitespace() {
                self.summary += word;
                if self.summary.len() >= 24 {
                    self.summary += "…";
                    break;
                }
                self.summary += " ";
            }
        }

        // clear chatbox
        self.chatbox.clear();

        // push prompt to ollama context messages
        self.context_messages.push(ChatMessage::user(prompt));
        let context_messages = self.context_messages.clone();

        // get ready for assistant response
        self.messages.push(Message::assistant(String::new()));

        // spawn a new thread to generate the completion
        let handle = self.flower.handle(); // recv'd by gui thread
        let ollama = ollama.clone();
        let stop_generation = self.stop_generating.clone();
        tokio::spawn(async move {
            handle.activate();
            let _ = request_completion(ollama, context_messages, &handle, stop_generation)
                .await
                .map_err(|e| {
                    log::error!("failed to request completion: {e}");
                    handle.error(e.to_string());
                });
        });
    }

    fn show_chatbox(
        &mut self,
        ui: &mut egui::Ui,
        is_max_height: bool,
        is_generating: bool,
        ollama: &Ollama,
    ) {
        if let Some(idx) = self.retry_message_idx.take() {
            self.chatbox = self.messages[idx].content.clone();
            self.messages.remove(idx + 1);
            self.messages.remove(idx);
            self.send_message(ollama);
        }

        if is_max_height {
            ui.add_space(8.0);
        }
        ui.horizontal_centered(|ui| {
            ui.add_enabled_ui(!is_generating, |ui| {
                if !is_max_height
                    && ui
                        .button("Send")
                        .on_disabled_hover_text("Please wait…")
                        .clicked()
                    && !is_generating
                {
                    self.send_message(ollama);
                }
            });
            ui.with_layout(
                Layout::left_to_right(Align::Center).with_main_justify(true),
                |ui| {
                    let Self { highlighter, .. } = self;
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
                        .height();
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
    }

    #[inline]
    pub fn flower_active(&self) -> bool {
        self.flower.is_active()
    }

    pub fn poll_flower(&mut self, modal: &mut Modal) {
        self.flower
            .extract(|progress| {
                self.messages.last_mut().unwrap().content += progress.as_str();
            })
            .finalize(|result| {
                let message = self.messages.last_mut().unwrap();

                if let Ok(content) = result {
                    message.content = content;
                } else if let Err(e) = result {
                    let msg = match e {
                        Compact::Panicked(e) => format!("Tokio task panicked: {e}"),
                        Compact::Suppose(e) => e,
                    };
                    // message.content = msg.clone();
                    message.is_error = true;
                    modal
                        .dialog()
                        .with_body(msg)
                        .with_title("Failed to generate completion!")
                        .with_icon(Icon::Error)
                        .open();
                }
                message.is_generating = false;
            });
    }

    pub fn last_message_contents(&self) -> Option<String> {
        for message in self.messages.iter().rev() {
            if message.content.is_empty() {
                continue;
            }
            return Some(if message.is_user {
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
                    ui.style().visuals.faint_bg_color
                } else {
                    ui.style().visuals.window_fill
                },
                Stroke::new(2.0, ui.style().visuals.window_stroke.color),
            );
            ui.painter().rect_stroke(
                rect.shrink(radius / 2.0 + 1.0),
                2.0,
                Stroke::new(2.0, Color32::DARK_GRAY),
            );
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        ollama: &Ollama,
        tts: SharedTts,
        stopped_speaking: bool,
        commonmark_cache: &mut CommonMarkCache,
    ) {
        let avail = ctx.available_rect();
        let max_height = avail.height() * 0.4 + 24.0;
        let chatbox_panel_height = self.chatbox_height + 24.0;
        let actual_chatbox_panel_height = chatbox_panel_height.min(max_height);
        let is_generating = self.flower_active();

        egui::TopBottomPanel::bottom("chatbox_panel")
            .exact_height(actual_chatbox_panel_height)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.show_chatbox(
                        ui,
                        chatbox_panel_height >= max_height,
                        is_generating,
                        ollama,
                    );
                });
            });

        let mut new_speaker: Option<usize> = None;
        egui::CentralPanel::default()
            .frame(Frame::central_panel(&ctx.style()).inner_margin(Margin {
                left: 16.0,
                right: 0.0,
                top: 0.0,
                bottom: 3.0,
            }))
            .show(ctx, |ui| {
                egui::ScrollArea::both()
                    .auto_shrink(false)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.add_space(16.0); // instead of centralpanel margin
                        for (i, message) in self.messages.iter_mut().enumerate() {
                            let prev_speaking = message.is_speaking;
                            if message.show(ui, commonmark_cache, tts.clone(), i) {
                                self.retry_message_idx = Some(i - 1);
                            }
                            if !prev_speaking && message.is_speaking {
                                new_speaker = Some(i);
                            }
                        }
                    });

                // stop generating button
                if is_generating {
                    self.stop_generating_button(
                        ui,
                        16.0,
                        pos2(
                            ui.cursor().min.x + 16.0,
                            avail.height() - 16.0 * 2.0 - actual_chatbox_panel_height,
                        ),
                    );
                }
            });

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
}
