import threading
from threading import Semaphore
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw, ImageTk
import time

# --- 1. Load Dataset ---
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0

# --- 2. Model Definition & Training ---
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_x, train_y, epochs=5, batch_size=32, verbose=1)

# --- 3. Extract Weights for Manual Forward Pass ---
dense1 = model.layers[1]
dense2 = model.layers[2]

W1, b1 = dense1.get_weights()
W2, b2 = dense2.get_weights()

# --- 4. Handwritten Digit Preprocessing ---
def preprocess_drawing(image_array, canvas_size):
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    
    coords = np.argwhere(image_array > 30)
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    padding = 10
    y_min = max(0, y_min - padding)
    y_max = min(canvas_size, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(canvas_size, x_max + padding)
    
    cropped = image_array[y_min:y_max, x_min:x_max]
    
    cropped_pil = Image.fromarray(cropped)
    cropped_pil = cropped_pil.resize((20, 20), Image.Resampling.LANCZOS)
    cropped = np.array(cropped_pil)
    
    final_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate Center of Mass
    total_mass = np.sum(cropped)
    if total_mass == 0:
        return None
    
    y_indices, x_indices = np.indices(cropped.shape)
    cy = np.sum(y_indices * cropped) / total_mass
    cx = np.sum(x_indices * cropped) / total_mass
    
    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))
    
    for i in range(20):
        for j in range(20):
            new_i = i + shift_y
            new_j = j + shift_x
            if 0 <= new_i < 28 and 0 <= new_j < 28:
                final_img[new_i, new_j] = cropped[i, j]
    
    if np.max(final_img) > 0:
        final_img = final_img.astype(float) / 255.0
    else:
        final_img = np.zeros((28, 28))
    
    return final_img

# --- 5. Threaded Forward Pass with Semaphores ---
def run_epoch(epoch_number, num_threads, log_callback, progress_callback=None):
    log_callback(f"\n========== EPOCH {epoch_number + 1} STARTED ==========")

    batch_x = train_x[epoch_number * 1000:(epoch_number + 1) * 1000]
    batch_y = train_y[epoch_number * 1000:(epoch_number + 1) * 1000]

    context = {
        "flat": None,
        "l1": None,
        "relu": None,
        "out": None
    }

    sem12 = Semaphore(0)
    sem23 = Semaphore(0)

    # Layer 1 Worker Threads
    def layer1():
        log_callback(f"Layer 1 started with {num_threads} worker threads")

        flat = batch_x.reshape(-1, 784)
        context["flat"] = flat
        context["l1"] = np.zeros((flat.shape[0], 128))

        chunk = int(np.ceil(flat.shape[0] / num_threads))
        workers = []

        def worker(start, end, idx):
            log_callback(f"  Worker {idx + 1}: samples [{start}:{end}]")
            context["l1"][start:end] = (
                np.dot(context["flat"][start:end], W1) + b1
            )

        for i in range(num_threads):
            start = i * chunk
            end = min(start + chunk, flat.shape[0])
            if start < end:
                t = threading.Thread(target=worker, args=(start, end, i))
                workers.append(t)
                t.start()

        for t in workers:
            t.join()

        log_callback("Layer 1 finished")
        sem12.release()

    # Layer 2 Activation (ReLU)
    def layer2():
        sem12.acquire()
        log_callback("Layer 2 started: ReLU activation")
        context["relu"] = np.maximum(0, context["l1"])
        log_callback("Layer 2 finished")
        sem23.release()

    # Layer 3 Classification (Softmax)
    def layer3():
        sem23.acquire()
        log_callback("Layer 3 started: Softmax output")
        logits = np.dot(context["relu"], W2) + b2
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        context["out"] = exp / np.sum(exp, axis=1, keepdims=True)
        log_callback("Layer 3 finished")

    # Start Layer threads
    t1 = threading.Thread(target=layer1)
    t2 = threading.Thread(target=layer2)
    t3 = threading.Thread(target=layer3)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    preds = np.argmax(context["out"], axis=1)
    accuracy = np.mean(preds == batch_y)

    log_callback(f"EPOCH {epoch_number + 1} ACCURACY = {accuracy * 100:.2f}%")
    
    if progress_callback:
        progress_callback(epoch_number + 1, accuracy)

# --- 6. GUI Application ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN + Threads + Semaphores - OS Project")
        self.root.geometry("900x900")
        self.root.minsize(850, 780)
        self.root.configure(bg="#1e1f22")

        try:
            self.root.tk.call("tk", "scaling", 1.25)
        except:
            pass

        # UI Styling Rules
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Main.TFrame", background="#1e1f22")
        style.configure("Card.TLabelframe", background="#2b2d31", foreground="white", borderwidth=2, relief="solid")
        style.configure("Card.TLabelframe.Label", background="#2b2d31", foreground="#00ffaa", font=("Segoe UI", 11, "bold"))
        style.configure("TLabel", background="#1e1f22", foreground="white", font=("Segoe UI", 11))
        style.configure("Card.TLabel", background="#2b2d31", foreground="white", font=("Segoe UI", 11))
        style.configure("Title.TLabel", background="#1e1f22", foreground="#00ffaa", font=("Segoe UI", 18, "bold"))
        style.configure("Result.TLabel", background="#1e1f22", foreground="#ffd166", font=("Segoe UI", 18, "bold"))
        style.configure("DrawResult.TLabel", background="#1e1f22", foreground="#ffd166", font=("Segoe UI", 18, "bold"))
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=10)
        style.configure("TNotebook", background="#1e1f22", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 12, "bold"), padding=[20, 10])

        # Main Frame Layout
        main = ttk.Frame(root, padding=14, style="Main.TFrame")
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=0)
        main.rowconfigure(1, weight=0)
        main.rowconfigure(2, weight=1)

        title = ttk.Label(main, text="CNN Forward Pass with Threads and Semaphores", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=8)

        # Tab 1: System Control
        self.tab1 = ttk.Frame(self.notebook, padding=15, style="Main.TFrame")
        self.notebook.add(self.tab1, text=" ⚙️ System Control ")

        self.tab1.columnconfigure(0, weight=1)
        self.tab1.rowconfigure(0, weight=0)
        self.tab1.rowconfigure(1, weight=0)
        self.tab1.rowconfigure(2, weight=1)

        control_frame = ttk.LabelFrame(self.tab1, text=" SYSTEM CONTROL ", padding=12, style="Card.TLabelframe")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        control_frame.columnconfigure(0, weight=0)
        control_frame.columnconfigure(1, weight=0)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=0)
        control_frame.columnconfigure(4, weight=0)

        ttk.Label(control_frame, text="Layer 1 Worker Threads:", style="Card.TLabel").grid(row=0, column=0, padx=8, pady=6, sticky="w")

        self.thread_entry = ttk.Entry(control_frame, width=10, font=("Segoe UI", 12))
        self.thread_entry.insert(0, "20")
        self.thread_entry.grid(row=0, column=1, padx=8, pady=6, sticky="w")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=5, length=150, mode='determinate')
        self.progress_bar.grid(row=0, column=2, padx=20, pady=6, sticky="ew")

        self.status_label = ttk.Label(control_frame, text="Ready", style="Card.TLabel")
        self.status_label.grid(row=0, column=3, padx=8, pady=6)

        self.run_button = ttk.Button(control_frame, text="Run 5 Epochs", style="Big.TButton", command=self.start_epochs)
        self.run_button.grid(row=0, column=4, padx=8, pady=6, sticky="e")

        log_frame = ttk.LabelFrame(self.tab1, text=" THREAD EXECUTION LOG ", padding=10, style="Card.TLabelframe")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 0))

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=10, bg="#111111", fg="#00ff99", insertbackground="white", font=("Consolas", 10), wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        # Tab 2: MNIST dataset inspection
        self.tab2 = ttk.Frame(self.notebook, padding=15, style="Main.TFrame")
        self.notebook.add(self.tab2, text=" 📊 MNIST Test Case ")

        self.tab2.columnconfigure(0, weight=1)
        self.tab2.rowconfigure(0, weight=0)
        self.tab2.rowconfigure(1, weight=0)
        self.tab2.rowconfigure(2, weight=1)

        test_card = ttk.LabelFrame(self.tab2, text=" SELECT IMAGE FROM MNIST TEST DATASET ", padding=15, style="Card.TLabelframe")
        test_card.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        test_card.columnconfigure(0, weight=1)
        test_card.columnconfigure(1, weight=0)
        test_card.columnconfigure(2, weight=0)

        ttk.Label(test_card, text="Image Index from 0 to 9999:", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=8)

        self.idx_entry = ttk.Entry(test_card, width=12, font=("Segoe UI", 13))
        self.idx_entry.insert(0, "45")
        self.idx_entry.grid(row=0, column=1, padx=8, pady=8)

        ttk.Button(test_card, text="Show & Predict", style="Big.TButton", command=self.show_test_image).grid(row=0, column=2, padx=8, pady=8)

        self.result_label = ttk.Label(self.tab2, text="Prediction: -    |    Actual: -", style="Result.TLabel", anchor="center")
        self.result_label.grid(row=1, column=0, sticky="ew", pady=12)

        image_area = ttk.Frame(self.tab2, style="Main.TFrame")
        image_area.grid(row=2, column=0, sticky="nsew")

        image_area.columnconfigure(0, weight=1)
        image_area.rowconfigure(0, weight=1)

        self.image_label = tk.Label(image_area, bg="#111111", width=320, height=320, bd=4, relief="ridge")
        self.image_label.grid(row=0, column=0, pady=15)

        # Tab 3: Free hand drawing
        self.tab3 = ttk.Frame(self.notebook, padding=15, style="Main.TFrame")
        self.notebook.add(self.tab3, text=" ✏️ Handwritten Drawing ")

        self.tab3.columnconfigure(0, weight=1)
        self.tab3.rowconfigure(0, weight=1)
        self.tab3.rowconfigure(1, weight=0)
        self.tab3.rowconfigure(2, weight=0)

        draw_card = ttk.LabelFrame(self.tab3, text=" DRAW A DIGIT WITH MOUSE ", padding=15, style="Card.TLabelframe")
        draw_card.grid(row=0, column=0, sticky="n", pady=(0, 10))

        self.canvas_size = 320

        self.canvas = tk.Canvas(draw_card, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=3, highlightbackground="#00ffaa")
        self.canvas.pack(padx=10, pady=10)

        self.last_x = None
        self.last_y = None
        self.pen_size = 18

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Base Image structure for PIL
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_image = ImageDraw.Draw(self.image)

        btn_frame = ttk.Frame(self.tab3, style="Main.TFrame")
        btn_frame.grid(row=1, column=0, sticky="ew", pady=10)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Predict Drawing", style="Big.TButton", command=self.predict_drawing).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(btn_frame, text="Clear Canvas", style="Big.TButton", command=self.clear_canvas).grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        self.draw_result = ttk.Label(self.tab3, text="Draw a digit, then click Predict Drawing", style="DrawResult.TLabel", anchor="center")
        self.draw_result.grid(row=2, column=0, sticky="ew", pady=10)

    def update_progress(self, epoch, accuracy):
        def update():
            self.progress_var.set(epoch)
            self.status_label.config(text=f"Epoch {epoch}/5 - {accuracy*100:.1f}%")
        self.root.after(0, update)

    def write_log(self, msg):
        def update():
            self.log.insert(tk.END, msg + "\n")
            self.log.see(tk.END)
        self.root.after(0, update)

    def start_epochs(self):
        self.log.delete("1.0", tk.END)
        self.progress_var.set(0)
        self.status_label.config(text="Running...")

        try:
            num_threads = int(self.thread_entry.get())
            if num_threads <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Please enter a valid positive number for threads.")
            return

        self.run_button.config(state="disabled")
        self.write_log("Starting threaded execution...")

        def runner():
            for i in range(5):
                run_epoch(i, num_threads, self.write_log, self.update_progress)
                time.sleep(0.1)

            self.write_log("\nALL EPOCHS FINISHED SUCCESSFULLY")
            self.root.after(0, lambda: self.run_button.config(state="normal"))
            self.root.after(0, lambda: self.status_label.config(text="Completed!"))

        threading.Thread(target=runner, daemon=True).start()

    def show_test_image(self):
        try:
            idx = int(self.idx_entry.get())
            if idx < 0 or idx > 9999:
                raise ValueError
        except:
            messagebox.showerror("Error", "Index must be between 0 and 9999.")
            return

        img = test_x[idx]

        flat = img.reshape(1, 784)
        l1 = np.dot(flat, W1) + b1
        relu = np.maximum(0, l1)
        logits = np.dot(relu, W2) + b2
        pred = np.argmax(logits)
        actual = test_y[idx]
        
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        confidence = probs[0][pred] * 100

        self.result_label.config(
            text=f"Predicted: {pred} ({confidence:.1f}%)    |    Actual: {actual}"
        )

        display = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(display)
        pil_img = pil_img.resize((320, 320), Image.Resampling.LANCZOS)
        
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_digit(self, event):
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return
        
        x1, y1 = self.last_x, self.last_y
        x2, y2 = event.x, event.y
        
        self.canvas.create_line(x1, y1, x2, y2, fill="white", width=self.pen_size, capstyle="round", smooth=True)
        self.draw_image.line([x1, y1, x2, y2], fill=255, width=self.pen_size)
        
        self.last_x = x2
        self.last_y = y2

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_image = ImageDraw.Draw(self.image)
        self.draw_result.config(text="Draw a digit, then click Predict Drawing")

    def predict_drawing(self):
        img_array = np.array(self.image)
        
        if np.max(img_array) == 0:
            self.draw_result.config(text="⚠️ Please draw a digit first!")
            return
        
        processed_img = preprocess_drawing(img_array, self.canvas_size)
        
        if processed_img is None:
            self.draw_result.config(text="⚠️ Could not detect digit! Please draw clearly.")
            return
        
        flat = processed_img.reshape(1, 784)
        l1 = np.dot(flat, W1) + b1
        relu = np.maximum(0, l1)
        logits = np.dot(relu, W2) + b2
        pred = np.argmax(logits)
        
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        confidence = probs[0][pred] * 100
        
        self.draw_result.config(
            text=f"Predicted Digit: {pred} (Confidence: {confidence:.1f}%)"
        )
        
        self.write_log(f"\n✏️ Drawing Prediction: {pred} (Confidence: {confidence:.1f}%)")
        self.write_log(f"Probabilities: {np.round(probs[0], 3)}")

# --- 7. Execution Entrypoint ---
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
