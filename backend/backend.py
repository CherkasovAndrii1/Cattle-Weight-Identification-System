import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------------------

import cv2
import numpy as np
import torch 
import torch.nn as nn 
import torchvision.models as models 
import tensorflow as tf 
from tensorflow.keras.saving import load_model as load_keras_model 
import time
import random
import json
import uuid
from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename

import albumentations as A
from albumentations.pytorch import ToTensorV2 
import segmentation_models_pytorch as smp
from flask_cors import CORS

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_jwt_extended import create_access_token, JWTManager, jwt_required, get_jwt_identity

from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app) 



APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
STATIC_FOLDER = os.path.join(APP_ROOT, 'static')
SEGMENTATION_OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'predictions')
KEYPOINTS_OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'keypoints_predictions')

TARGET_SIZE_KP = (224, 224) 
NUM_KEYPOINTS = 9
KEYPOINT_MODEL_PATH = 'best_keypoints_model_9pts_limited.keras' 


SEGMENTATION_MODEL_PATH = 'best_cattle_segmentation_model_6.keras'
NUM_CLASSES_SEG = 3
TARGET_SIZE_SEG = (512, 512) 

WEIGHT_MODEL_PATH = 'best_enhanced_triple_cattle_weight_model_res18_100ep_2600p.pth' 
TARGET_SIZE_WEIGHT = (224, 224) 

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTATION_OUTPUT_FOLDER'] = SEGMENTATION_OUTPUT_FOLDER
app.config['KEYPOINTS_OUTPUT_FOLDER'] = KEYPOINTS_OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEGMENTATION_OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['KEYPOINTS_OUTPUT_FOLDER'], exist_ok=True)

# Device configuration (для PyTorch)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-super-secret-key-replace-me') 
# ------------------------------------

db = SQLAlchemy(app)      
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)     
jwt = JWTManager(app)     
# --------------------------------


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False) 
    password_hash = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.email}')"

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    processing_type = db.Column(db.String(50), nullable=False) 
    original_filename = db.Column(db.String(200))
    result_image_url = db.Column(db.String(500))
    result_data = db.Column(db.Text, nullable=True) 
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('uploads', lazy=True))

    def __repr__(self):
        return f"<UploadHistory {self.id} User: {self.user_id} Type: {self.processing_type}>"

    def to_dict(self):
        """Преобразует объект в словарь для JSON ответа."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "processing_type": self.processing_type,
            "original_filename": self.original_filename,
            "result_image_url": self.result_image_url,
            "result_data": json.loads(self.result_data) if self.result_data else None,
            "uploaded_at": self.uploaded_at.isoformat()
        }




class EnhancedTripleInputCattleWeightCNN(nn.Module):
    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
       super().__init__()
       self.orig_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
       num_ftrs_orig = self.orig_backbone.fc.in_features
       self.orig_backbone.fc = nn.Identity()
       self.orig_bn = nn.BatchNorm1d(num_ftrs_orig)

       self.seg_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
       num_ftrs_seg = self.seg_backbone.fc.in_features
       self.seg_backbone.fc = nn.Identity()
       self.seg_bn = nn.BatchNorm1d(num_ftrs_seg)

       keypoints_input_dim = num_keypoints * 2
       self.keypoints_fc = nn.Sequential(
           nn.Linear(keypoints_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
           nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True)
       )
       keypoints_feature_dim = 256

       combined_input_dim = num_ftrs_orig + num_ftrs_seg + keypoints_feature_dim
       self.combined_fc = nn.Sequential(
           nn.BatchNorm1d(combined_input_dim),
           nn.Linear(combined_input_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
           nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
           nn.Linear(128, 1)
       )

    def forward(self, orig_img: torch.Tensor, seg_img: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
       orig_features = self.orig_bn(self.orig_backbone(orig_img))
       seg_features = self.seg_bn(self.seg_backbone(seg_img))
       keypoints_features = self.keypoints_fc(keypoints)
       combined_features = torch.cat((orig_features, seg_features, keypoints_features), dim=1)
       weight = self.combined_fc(combined_features)
       return weight

def create_segmentation_model_pytorch(num_classes=NUM_CLASSES_SEG):
    try:
        model_instance = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        return model_instance
    except Exception as e:
        print(f"Error creating PyTorch segmentation model structure: {e}")
        raise


segmentation_model = None
keypoint_model = None
weight_prediction_model = None


print("--- Загрузка моделей ---")

try:
    segmentation_model = create_segmentation_model_pytorch(NUM_CLASSES_SEG) # Создаем структуру
    if os.path.exists(SEGMENTATION_MODEL_PATH):
        segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE))
        segmentation_model.to(DEVICE)
        segmentation_model.eval()
        print(f"Segmentation model (PyTorch) loaded successfully from {SEGMENTATION_MODEL_PATH} on {DEVICE}")
    else:
        print(f"Warning: Segmentation model file not found at {SEGMENTATION_MODEL_PATH}.")
        segmentation_model = None
except Exception as e:
    print(f"Error loading PyTorch segmentation model: {e}")
    segmentation_model = None


try:
    if os.path.exists(KEYPOINT_MODEL_PATH):
        keypoint_model = load_keras_model(KEYPOINT_MODEL_PATH)
        print(f"Keypoint model (Keras) loaded successfully from {KEYPOINT_MODEL_PATH}")
    else:
        print(f"Warning: Keypoint model file not found at {KEYPOINT_MODEL_PATH}.")
        keypoint_model = None
except Exception as e:
    print(f"Error loading keypoint model: {e}")
    import traceback; traceback.print_exc()
    keypoint_model = None

try:
    weight_prediction_model = EnhancedTripleInputCattleWeightCNN(num_keypoints=NUM_KEYPOINTS, pretrained=False).to(DEVICE)
    if os.path.exists(WEIGHT_MODEL_PATH):
        weight_prediction_model.load_state_dict(torch.load(WEIGHT_MODEL_PATH, map_location=DEVICE))
        weight_prediction_model.eval()
        print(f"Weight prediction model loaded successfully from {WEIGHT_MODEL_PATH}")
    else:
        print(f"Warning: Weight prediction model file not found at {WEIGHT_MODEL_PATH}")
        weight_prediction_model = None
except Exception as e:
    print(f"Error loading weight prediction model: {e}")
    weight_prediction_model = None
print("--- Загрузка моделей завершена ---")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_pytorch_transforms(target_size: tuple[int, int], is_train: bool = False) -> A.Compose:
    return A.Compose([
        A.Resize(height=target_size[1], width=target_size[0]),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

def predict_segmentation(image_rgb, target_size_seg=TARGET_SIZE_SEG):
    if not isinstance(segmentation_model, torch.nn.Module):
        print("Warning: PyTorch segmentation model not available or loaded incorrectly. Skipping segmentation.")
        return None, None, None, "Модель сегментации PyTorch не загружена или неверного типа."
    try:
        original_height, original_width = image_rgb.shape[:2]
        transform = A.Compose([
            A.Resize(height=target_size_seg[1], width=target_size_seg[0]),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
        augmented = transform(image=image_rgb); processed_image = augmented['image']
        image_tensor = torch.from_numpy(np.transpose(processed_image, (2, 0, 1))).float().unsqueeze(0)
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE); output = segmentation_model(image_tensor)
            if len(output.shape) == 4:
                 pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            elif len(output.shape) == 3:
                 pred_mask = output.squeeze().cpu().numpy().astype(np.uint8)
            else: raise ValueError(f"Неожиданная форма выхода модели сегментации: {output.shape}")
        resized_mask = cv2.resize(pred_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        return resized_mask, original_height, original_width, None
    except Exception as e:
        print(f"Error in segmentation prediction: {str(e)}"); import traceback; traceback.print_exc()
        return None, None, None, f"Ошибка сегментации: {str(e)}"

def apply_mask_to_image(image_bgr, mask_original_size):
    if mask_original_size is None or image_bgr is None: return image_bgr
    try:
        CATTLE_CLASS_INDEX = 1
        binary_mask = (mask_original_size == CATTLE_CLASS_INDEX).astype(np.uint8)
        if np.sum(binary_mask) == 0:
             print("Warning: Segmentation mask for 'cattle' class is empty.")
             return np.zeros_like(image_bgr) 
        mask_3channel = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
        segmented_image_bgr = cv2.bitwise_and(image_bgr, mask_3channel)
        return segmented_image_bgr
    except Exception as e:
        print(f"Ошибка применения маски: {e}"); return image_bgr

def rescale_keypoints(keypoints_pred_scaled_flat, orig_width, orig_height, target_size_kp=TARGET_SIZE_KP):
    try:
        if keypoints_pred_scaled_flat is None: return None, "Нет входных точек"
        if orig_width <= 0 or orig_height <= 0: return None, "Неверные оригинальные размеры"
        if target_size_kp[0] <= 0 or target_size_kp[1] <= 0: return None, "Неверные размеры входа модели точек"
        kp_shape = keypoints_pred_scaled_flat.shape; num_elements_pred = keypoints_pred_scaled_flat.size; expected_elements = NUM_KEYPOINTS * 2
        if len(kp_shape) == 2 and kp_shape[0] == 1: keypoints_flat = keypoints_pred_scaled_flat[0]
        elif len(kp_shape) == 1: keypoints_flat = keypoints_pred_scaled_flat
        else: return None, f"Неожиданная форма входных точек: {kp_shape}"
        if keypoints_flat.size > expected_elements:
             print(f"Предупреждение: Предсказано больше точек ({keypoints_flat.size // 2}), чем ожидалось ({NUM_KEYPOINTS}). Используются первые {NUM_KEYPOINTS}.")
             keypoints_flat = keypoints_flat[:expected_elements]
        elif keypoints_flat.size < expected_elements: return None, f"Предсказано меньше точек ({keypoints_flat.size // 2}), чем ожидалось ({NUM_KEYPOINTS})."
        scaled_keypoints_xy = keypoints_flat.reshape(NUM_KEYPOINTS, 2)
        scale_x = orig_width / target_size_kp[0]; scale_y = orig_height / target_size_kp[1]
        rescaled_keypoints = scaled_keypoints_xy * np.array([scale_x, scale_y])
        rescaled_keypoints[:, 0] = np.clip(rescaled_keypoints[:, 0], 0, orig_width - 1)
        rescaled_keypoints[:, 1] = np.clip(rescaled_keypoints[:, 1], 0, orig_height - 1)
        return rescaled_keypoints, None
    except Exception as e:
        print(f"Ошибка масштабирования точек: {e}"); import traceback; traceback.print_exc()
        return None, f"Ошибка масштабирования точек: {str(e)}"

def predict_keypoints_keras(image_bgr, model, target_size=TARGET_SIZE_KP):
    if model is None: return None, "Модель ключевых точек не загружена"
    try:
        orig_height, orig_width = image_bgr.shape[:2]
        image_resized = cv2.resize(image_bgr, target_size)
        image_input = image_resized / 255.0 # Ожидаем BGR [0,1]
        image_input = np.expand_dims(image_input, axis=0)
        if not isinstance(model, tf.keras.Model):
             return None, "Загруженный объект для keypoint_model не является моделью Keras."
        keypoints_pred_scaled_flat = model.predict(image_input, verbose=0)
        keypoints_original, error = rescale_keypoints(keypoints_pred_scaled_flat, orig_width, orig_height, target_size)
        return keypoints_original, error
    except Exception as e:
        print(f"Ошибка при предсказании точек Keras: {e}"); import traceback; traceback.print_exc()
        return None, f"Ошибка предсказания точек Keras: {str(e)}"

def predict_weight_combined(
    orig_image_bgr: np.ndarray, weight_model: nn.Module, keypoint_model_keras,
    segmentation_model_pytorch: nn.Module, device: torch.device,
    target_size_kp: tuple[int, int] = TARGET_SIZE_KP, target_size_seg: tuple[int, int] = TARGET_SIZE_SEG,
    target_size_weight: tuple[int, int] = TARGET_SIZE_WEIGHT
) -> tuple[float | None, str | None]:
    if not isinstance(weight_model, torch.nn.Module): return None, "Модель предсказания веса не загружена или неверного типа."
    if keypoint_model_keras is None : return None, "Модель ключевых точек Keras не загружена." # Явная проверка
    start_time_total = time.time(); processing_times = {}
    try:
       
        kp_start = time.time()
        keypoints_xy, kp_error = predict_keypoints_keras(orig_image_bgr, keypoint_model_keras, target_size_kp)
        processing_times['keypoints_prediction'] = time.time() - kp_start
        if kp_error: return None, f"Ошибка ключевых точек: {kp_error}"
        if keypoints_xy is None: return None, "Не удалось получить ключевые точки (None)."
        if keypoints_xy.shape != (NUM_KEYPOINTS, 2): return None, f"Неверная форма ключевых точек после предсказания: {keypoints_xy.shape}"

       
        seg_start = time.time(); orig_image_rgb = cv2.cvtColor(orig_image_bgr, cv2.COLOR_BGR2RGB)
        
        seg_mask_original_size, h, w, seg_error = predict_segmentation(orig_image_rgb, target_size_seg)
        processing_times['segmentation_prediction'] = time.time() - seg_start
        

        
        mask_apply_start = time.time()
        if seg_mask_original_size is not None:
            segmented_image_bgr = apply_mask_to_image(orig_image_bgr, seg_mask_original_size)
            if seg_error: print(f"Warning: Segmentation error occurred ({seg_error}), but mask was applied.")
        else:
            print(f"Предупреждение: Сегментация не выполнена или не удалась ({seg_error}). Используется оригинальное изображение для второго входа модели веса.")
            segmented_image_bgr = orig_image_bgr.copy() # Используем копию оригинала
        processing_times['mask_application'] = time.time() - mask_apply_start

        
        prep_start = time.time(); transform_weight = get_pytorch_transforms(target_size_weight)
        orig_image_rgb_for_weight = cv2.cvtColor(orig_image_bgr, cv2.COLOR_BGR2RGB)
        segmented_image_rgb_for_weight = cv2.cvtColor(segmented_image_bgr, cv2.COLOR_BGR2RGB)
        transformed_orig = transform_weight(image=orig_image_rgb_for_weight); orig_tensor = transformed_orig['image'].unsqueeze(0).to(device)
        transformed_seg = transform_weight(image=segmented_image_rgb_for_weight); seg_tensor = transformed_seg['image'].unsqueeze(0).to(device)
        orig_height, orig_width = orig_image_bgr.shape[:2]; normalized_keypoints_flat = []
        for x, y in keypoints_xy: norm_x = x / orig_width; norm_y = y / orig_height; normalized_keypoints_flat.extend([norm_x, norm_y])
        normalized_keypoints_flat = [np.clip(v, 0.0, 1.0) for v in normalized_keypoints_flat]
        if len(normalized_keypoints_flat) != NUM_KEYPOINTS * 2: return None, f"Неверное количество нормализованных точек: {len(normalized_keypoints_flat)}"
        keypoints_tensor = torch.tensor(normalized_keypoints_flat, dtype=torch.float32).unsqueeze(0).to(device)
        processing_times['preprocessing'] = time.time() - prep_start

        
        weight_start = time.time(); weight_model.eval()
        with torch.no_grad(): predicted_weight = weight_model(orig_tensor, seg_tensor, keypoints_tensor).item()
        processing_times['weight_prediction'] = time.time() - weight_start

        total_time = time.time() - start_time_total
        print(f"Время обработки для предсказания веса: {total_time:.3f}s"); print(f"Детализация времени: {processing_times}")
        return float(predicted_weight), None
    except Exception as e:
        print(f"Ошибка в predict_weight_combined: {e}"); import traceback; traceback.print_exc()
        return None, f"Внутренняя ошибка при предсказании веса: {str(e)}"


def save_segmentation_overlay(original_img_rgb, pred_mask, original_height, original_width, filename):
    if pred_mask is None:
        print("Error: Cannot save segmentation overlay, prediction mask is None.")
        return None
    try:
        base, ext = os.path.splitext(filename); safe_filename = f'segmentation_{secure_filename(base)}_{str(uuid.uuid4())[:8]}.png'
        output_path_absolute = os.path.join(app.config['SEGMENTATION_OUTPUT_FOLDER'], safe_filename)
        resized_mask = pred_mask 
        overlay_rgb = original_img_rgb.copy(); colors = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]} # Убедитесь, что индексы верны!
        for class_id, color_rgb in colors.items():
            if class_id >= 0:
                 overlay_rgb[resized_mask == class_id] = list(color_rgb)
        alpha = 0.4; final_overlay = cv2.addWeighted(overlay_rgb, alpha, original_img_rgb, 1 - alpha, 0)
        overlay_bgr = cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR); success = cv2.imwrite(output_path_absolute, overlay_bgr)
        if success and os.path.exists(output_path_absolute):
            print(f"Segmentation overlay saved: {output_path_absolute}")
            relative_path_for_url = os.path.join('predictions', safe_filename).replace("\\", "/")
            return relative_path_for_url
        else: print(f"Error saving/verifying segmentation file: {output_path_absolute}"); return None
    except Exception as e: print(f"Error saving segmentation overlay: {str(e)}"); import traceback; traceback.print_exc(); return None

def draw_and_save_keypoints_visualization(image_bgr, keypoints, filename):
    if keypoints is None:
         print("Error: Cannot draw keypoints visualization, keypoints array is None.")
         return None, "Keypoints array is None, cannot visualize."
    try:
        base, ext = os.path.splitext(filename); unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"keypoints_{secure_filename(base)}_{unique_id}.jpg"
        output_path_absolute = os.path.join(app.config['KEYPOINTS_OUTPUT_FOLDER'], safe_filename)
        image_with_keypoints = image_bgr.copy()
        color_map = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,128), (0,128,128), (128,128,0)]
        radius = max(5, int(min(image_bgr.shape[:2]) * 0.01)); thickness = -1; contour_thickness = 2; font_scale = 0.6; font_thickness = 1
        for i, point in enumerate(keypoints):
             x, y = int(point[0]), int(point[1])
             if 0 <= x < image_with_keypoints.shape[1] and 0 <= y < image_with_keypoints.shape[0]:
                 color_idx = i % len(color_map); color = color_map[color_idx]; center = (x, y)
                 cv2.circle(image_with_keypoints, center, radius + contour_thickness,(255, 255, 255), contour_thickness)
                 cv2.circle(image_with_keypoints, center, radius, color, thickness)
                 text_pos = (x + radius // 2, y - radius // 2)
                 cv2.putText(image_with_keypoints, str(i+1), (text_pos[0]+1, text_pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
                 cv2.putText(image_with_keypoints, str(i+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        success = cv2.imwrite(output_path_absolute, image_with_keypoints)
        if success and os.path.exists(output_path_absolute):
            print(f"Keypoint visualization saved: {output_path_absolute}")
            relative_path_for_url = os.path.join('keypoints_predictions', safe_filename).replace("\\", "/")
            return relative_path_for_url, None
        else: error_msg = f"Ошибка сохранения файла визуализации точек: {output_path_absolute}"; print(error_msg); return None, error_msg
    except Exception as e: import traceback; error_msg = f"Ошибка при визуализации точек: {str(e)}"; print(error_msg); traceback.print_exc(); return None, error_msg

def get_segmentation_stats(mask, original_height, original_width):
    if mask is None: return None
    try:
        resized_mask = mask; total_pixels = resized_mask.size;
        if total_pixels == 0: return None
        stats = {}; class_names = {0: "background", 1: "cattle", 2: "sticker"} # Убедитесь что индексы и имена верны
        for class_id, name in class_names.items():
             pixels = np.sum(resized_mask == class_id); percentage = (pixels / total_pixels) * 100 if total_pixels > 0 else 0;
             stats[f"{name}_pixels"] = int(pixels); stats[f"{name}_percentage"] = float(f"{percentage:.2f}")
        return stats
    except Exception as e: print(f"Error calculating stats: {str(e)}"); return None


def process_keypoints_and_save(image_bgr: np.ndarray, original_filename: str) -> tuple[np.ndarray | None, str | None, str | None]:

    if keypoint_model is None:
        return None, None, "Keypoint model is not loaded"

    try:
        keypoints_array, error_pred = predict_keypoints_keras(image_bgr, keypoint_model, target_size=TARGET_SIZE_KP)
        if error_pred:
            return None, None, f"Keypoint prediction failed: {error_pred}"
        if keypoints_array is None: # На всякий случай
             return None, None, "Keypoint prediction returned None without error."
        if keypoints_array.shape != (NUM_KEYPOINTS, 2):
             return None, None, f"Keypoint prediction returned invalid shape: {keypoints_array.shape}"


        relative_path_for_url, error_vis = draw_and_save_keypoints_visualization(image_bgr, keypoints_array, original_filename)
        if error_vis:

            return keypoints_array, None, f"Failed to save keypoint visualization: {error_vis}"

        return keypoints_array, relative_path_for_url, None # Успех

    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, f"Internal error during keypoint processing/saving: {str(e)}"
# ----------------------------------------------------




@app.route('/api/v1/segmentation', methods=['POST'])
@jwt_required() 
def segmentation_predict_route():
    user_id = get_jwt_identity()
    start_time = time.time()
    processing_time_str = "N/A"; file_path = None; status_code = 200

    if 'file' not in request.files: return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"success": False, "error": "No selected file"}), 400
    if not (file and allowed_file(file.filename)): return jsonify({"success": False, "error": "File type not allowed"}), 400

    original_filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_user{user_id}_{original_filename}")

    try:
        file.save(file_path); print(f"File saved by user {user_id} to {file_path}")
        image_bgr = cv2.imread(file_path)
        if image_bgr is None: return jsonify({"success": False, "error": "Could not read image"}), 400
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pred_mask, original_height, original_width, error = predict_segmentation(image_rgb, target_size_seg=TARGET_SIZE_SEG)

        if segmentation_model is None or not isinstance(segmentation_model, torch.nn.Module):
             if error:
                 print(f"Segmentation prediction function reported error: {error}")
                 
                 pred_mask = None 
             else: 
                 print("Segmentation model not loaded or not a PyTorch model.")
                 pred_mask = None

        elif error: 
             return jsonify({"success": False, "error": f"Segmentation prediction failed: {error}"}), 500

        processing_time_s = time.time() - start_time
        processing_time_str = f"{processing_time_s:.3f}s"

        output_rel_path = save_segmentation_overlay(image_rgb, pred_mask, original_height, original_width, original_filename)
        if output_rel_path is None:
             print("Warning: Failed to save segmentation overlay.")

        stats = get_segmentation_stats(pred_mask, original_height, original_width)
        image_url = None
        if output_rel_path: 
            try:
                image_url = url_for('static', filename=output_rel_path, _external=True)
                print(f"Generated Segmentation URL: {image_url}")
            except Exception as url_e:
                print(f"Error generating URL: {url_e}")

        try:
            history_entry = UploadHistory(
                user_id=user_id, processing_type='segmentation', original_filename=original_filename,
                result_image_url=image_url, result_data=json.dumps(stats) if stats else None
            )
            db.session.add(history_entry); db.session.commit()
            print(f"Saved segmentation history for user {user_id}")
        except Exception as db_e:
            db.session.rollback(); print(f"!!! DATABASE ERROR saving history for user {user_id}: {db_e}")

        return jsonify({"success": True, "image_url": image_url, "stats": stats, "processing_time": processing_time_str}), status_code
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal error: {str(e)}"}), 500
    finally:
        if file_path and os.path.exists(file_path):
             try: os.remove(file_path); print(f"Removed temp file: {file_path}")
             except OSError as err: print(f"Error removing file {file_path}: {err}")


@app.route('/api/v1/keypoints', methods=['POST'])
@jwt_required()
def keypoints_predict_route():
    user_id = get_jwt_identity()
    start_time = time.time(); processing_time_str = "N/A"; upload_path = None; status_code = 200

    if 'file' not in request.files: return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"success": False, "error": "No selected file"}), 400
    if not (file and allowed_file(file.filename)): return jsonify({"success": False, "error": "File type not allowed"}), 400

    original_filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{str(uuid.uuid4())}_keypoints_user{user_id}_{original_filename}")

    try:
        file_bytes = file.read()
        if not file_bytes:
             return jsonify({"success": False, "error": "Could not read file bytes"}), 400
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None: return jsonify({"success": False, "error": "Could not decode image from bytes"}), 400
        print(f"Image decoded successfully for user {user_id}")

        keypoints_array, relative_viz_path, error_process = process_keypoints_and_save(image_bgr, original_filename)

        if error_process:
             if keypoints_array is None: 
                 print(f"Error processing keypoints (fatal): {error_process}")
                 err_code = 503 if "model is not loaded" in error_process else 500
                 return jsonify({"success": False, "error": error_process}), err_code
             else: 
                 print(f"Warning processing keypoints (visualization failed): {error_process}")


        image_url = None
        if relative_viz_path:
            try:
                image_url = url_for('static', filename=relative_viz_path, _external=True)
                print(f"Generated Keypoints URL: {image_url}")
            except Exception as url_e:
                print(f"Error generating URL for keypoints visualization: {url_e}")

        processing_time_s = time.time() - start_time; processing_time_str = f"{processing_time_s:.3f}s"


        try:
             history_entry = UploadHistory(
                 user_id=user_id, processing_type='keypoints', original_filename=original_filename,
                 result_image_url=image_url, # Может быть None
                 result_data=json.dumps({"keypoints": keypoints_array.tolist()}) if keypoints_array is not None else None
             )
             db.session.add(history_entry); db.session.commit()
             print(f"Saved keypoints history for user {user_id}")
        except Exception as db_e:
             db.session.rollback(); print(f"!!! DATABASE ERROR saving keypoints history for user {user_id}: {db_e}")

        result = {
            "success": True, "message": "Точки успешно определены", "image_url": image_url,
            "keypoints": keypoints_array.tolist() if keypoints_array is not None else [],
            "processing_time": processing_time_str,
            "visualization_error": error_process if relative_viz_path is None and keypoints_array is not None else None
        }
        return jsonify(result), status_code
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": f"Внутренняя ошибка сервера: {str(e)}"}), 500
    finally:

        pass


@app.route('/api/v1/weight', methods=['POST'])
@jwt_required()
def weight_predict_route():
    user_id = get_jwt_identity()
    start_time = time.time(); processing_time_str = "N/A"; upload_path = None; status_code = 200; predicted_weight = None; weight_image_url = None; keypoints_image_url = None; kp_process_error = None

    if 'file' not in request.files: return jsonify({"success": False, "error": "No 'file' part in the request"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"success": False, "error": "No selected file"}), 400
    if not allowed_file(file.filename): return jsonify({"success": False, "error": "File type not allowed"}), 400

    if not isinstance(weight_prediction_model, torch.nn.Module): # Проверяем тип модели веса
         return jsonify({"success": False, "error": "Weight prediction model is not loaded or is not a PyTorch model"}), 503

    original_filename = secure_filename(file.filename)

    file_bytes = file.read()
    if not file_bytes: return jsonify({"success": False, "error": "Could not read file bytes"}), 400
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None: return jsonify({"success": False, "error": "Could not decode image from bytes"}), 400
    print(f"Image decoded successfully by user {user_id} for weight prediction")

    try:
        kp_process_start = time.time()
        keypoints_xy, relative_kp_viz_path, kp_process_error = process_keypoints_and_save(image_bgr, original_filename)
        print(f"Keypoint processing and saving took: {time.time() - kp_process_start:.3f}s")

        if keypoints_xy is None: 
            print(f"Failed to get keypoints: {kp_process_error}")
            err_code = 503 if "model is not loaded" in (kp_process_error or "") else 500
            return jsonify({"success": False, "error": "Failed to obtain keypoints.", "details": kp_process_error}), err_code

        if relative_kp_viz_path is None and kp_process_error: 
             print(f"Warning: Keypoints obtained, but visualization failed: {kp_process_error}")

        if relative_kp_viz_path:
            try:
                keypoints_image_url = url_for('static', filename=relative_kp_viz_path, _external=True)
                print(f"Generated Keypoints Visualization URL: {keypoints_image_url}")
            except Exception as url_e:
                print(f"Error generating URL for keypoints visualization: {url_e}")
                keypoints_image_url = None


        seg_start = time.time(); segmented_image_bgr = None; seg_error = None
        if isinstance(segmentation_model, torch.nn.Module):
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            seg_mask_original_size, h, w, seg_error = predict_segmentation(image_rgb, target_size_seg=TARGET_SIZE_SEG)
            if seg_error: print(f"Warning: Segmentation failed: {seg_error}. Using original image."); segmented_image_bgr = image_bgr.copy()
            elif seg_mask_original_size is not None: segmented_image_bgr = apply_mask_to_image(image_bgr, seg_mask_original_size)
            else: print("Warning: Segmentation returned None without error."); segmented_image_bgr = image_bgr.copy()
        else:
             print("Warning: Segmentation model not loaded or not a PyTorch model. Using original image.")
             segmented_image_bgr = image_bgr.copy() # Используем копию оригинала, если сегментации нет
             seg_error = "Segmentation model not loaded or not PyTorch" # Устанавливаем ошибку для информации
        print(f"Internal segmentation time: {time.time() - seg_start:.3f}s (Error: {seg_error})")

        prep_start = time.time(); transform_weight = get_pytorch_transforms(TARGET_SIZE_WEIGHT)
        orig_image_rgb_for_weight = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB); segmented_image_rgb_for_weight = cv2.cvtColor(segmented_image_bgr, cv2.COLOR_BGR2RGB)
        transformed_orig = transform_weight(image=orig_image_rgb_for_weight); orig_tensor = transformed_orig['image'].unsqueeze(0).to(DEVICE)
        transformed_seg = transform_weight(image=segmented_image_rgb_for_weight); seg_tensor = transformed_seg['image'].unsqueeze(0).to(DEVICE)
        orig_height, orig_width = image_bgr.shape[:2]; normalized_keypoints_flat = []
        for x, y in keypoints_xy: norm_x = x / orig_width; norm_y = y / orig_height; normalized_keypoints_flat.extend([norm_x, norm_y])
        normalized_keypoints_flat = [np.clip(v, 0.0, 1.0) for v in normalized_keypoints_flat]
        if len(normalized_keypoints_flat) != NUM_KEYPOINTS * 2: return jsonify({"success": False, "error": "Incorrect number of normalized keypoints"}), 500
        keypoints_tensor = torch.tensor(normalized_keypoints_flat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        print(f"Preprocessing time: {time.time() - prep_start:.3f}s")

        weight_start = time.time(); weight_prediction_model.eval()
        with torch.no_grad(): predicted_weight = weight_prediction_model(orig_tensor, seg_tensor, keypoints_tensor).item()
        print(f"Weight prediction time: {time.time() - weight_start:.3f}s")

        base_image_for_weight_text = None
        if keypoints_image_url: 
             try:
                 static_prefix = '/static/';
                 if static_prefix in keypoints_image_url:
                     rel_path = keypoints_image_url.split(static_prefix)[1]
                     keypoints_image_phys_path = os.path.join(app.static_folder, rel_path)
                     if os.path.exists(keypoints_image_phys_path):
                         base_image_for_weight_text = cv2.imread(keypoints_image_phys_path)
                         if base_image_for_weight_text is None: # Проверка, если imread не удался
                              print(f"Warning: Failed to read keypoints image at {keypoints_image_phys_path}")
                         else:
                              print("Using keypoints visualization as base for weight text.")
             except Exception as read_kp_img_e: print(f"Could not process keypoints image path for weight text base: {read_kp_img_e}")
        if base_image_for_weight_text is None: base_image_for_weight_text = image_bgr.copy(); print("Using original image as base for weight text.")

        try:
            base, ext = os.path.splitext(original_filename); unique_id = str(uuid.uuid4())[:8]
            safe_filename = f"weight_{secure_filename(base)}_{unique_id}.jpg"
            weight_image_path = os.path.join(app.config['KEYPOINTS_OUTPUT_FOLDER'], safe_filename)
            font = cv2.FONT_HERSHEY_SIMPLEX; text = f"Вес: {round(predicted_weight, 1)} кг"; text_pos = (20, 40)
            (text_width, text_height), _ = cv2.getTextSize(text, font, 1.2, 2)
            cv2.rectangle(base_image_for_weight_text, (text_pos[0] - 5, text_pos[1] - text_height - 5), (text_pos[0] + text_width + 5, text_pos[1] + 5), (0, 0, 0), -1)
            cv2.putText(base_image_for_weight_text, text, text_pos, font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            success_write = cv2.imwrite(weight_image_path, base_image_for_weight_text)
            if not success_write: print(f"Error: Failed to write weight visualization image to {weight_image_path}")
            else: print(f"Weight visualization saved at: {weight_image_path}")

            if success_write and os.path.exists(weight_image_path):
                 relative_path = os.path.relpath(weight_image_path, app.static_folder).replace("\\", "/")
                 try: weight_image_url = url_for('static', filename=relative_path, _external=True); print(f"Generated Weight Image URL: {weight_image_url}")
                 except Exception as url_e: print(f"Error generating weight image URL: {url_e}")
        except Exception as viz_e: print(f"Error creating weight visualization: {viz_e}"); import traceback; traceback.print_exc()


        processing_time_s = time.time() - start_time; processing_time_str = f"{processing_time_s:.3f}s"

        try:
             result_weight_data = {"estimated_weight_kg": round(predicted_weight, 2)}
             history_entry = UploadHistory(
                 user_id=user_id, processing_type='weight', original_filename=original_filename,
                 result_image_url=weight_image_url, result_data=json.dumps(result_weight_data)
             )
             db.session.add(history_entry); db.session.commit()
             print(f"Saved weight history for user {user_id}")
        except Exception as db_e:
             db.session.rollback(); print(f"!!! DATABASE ERROR saving weight history for user {user_id}: {db_e}")

        result = {
            "success": True, "estimated_weight_kg": round(predicted_weight, 2),
            "processing_time": processing_time_str,
            "segmentation_error": seg_error, 
            "keypoints_obtained_successfully": keypoints_xy is not None,
            "image_url": weight_image_url, 
            "keypoints_image_url": keypoints_image_url, 
            "keypoint_processing_error": kp_process_error 
        }
        return jsonify(result), status_code

    except Exception as e:
        import traceback; traceback.print_exc()
        error_message = f"Internal server error during weight prediction: {str(e)}"
        if kp_process_error: error_message += f" | Keypoint Processing Error: {kp_process_error}"
        return jsonify({"success": False, "error": error_message}), 500
    finally:
        pass
# ----------------------------------------------------

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/v1/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400
    email = data.get('email'); password = data.get('password')
    existing_user = User.query.filter_by(email=email).first()
    if existing_user: return jsonify({"error": "Email already registered"}), 409
    try:
        new_user = User(email=email); new_user.set_password(password)
        db.session.add(new_user); db.session.commit(); print(f"User registered: {email}")
        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        db.session.rollback(); print(f"Error during registration: {e}")
        return jsonify({"error": "Registration failed due to server error"}), 500

@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400
    email = data.get('email'); password = data.get('password')
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=str(user.id))
        print(f"User logged in: {email}")
        return jsonify(access_token=access_token), 200
    else:
        print(f"Failed login attempt for: {email}")
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/v1/users/me', methods=['GET'])
@jwt_required() 
def get_current_user():
    current_user_id = get_jwt_identity(); user = User.query.get(current_user_id)
    if not user: return jsonify({"error": "User not found"}), 404
    user_info = {"id": user.id, "email": user.email}
    return jsonify(user_info), 200


@app.route('/api/v1/history', methods=['GET'])
@jwt_required() 
def get_history():
    user_id = get_jwt_identity()
    print(f"Fetching history for user_id: {user_id}")
    try:
        history_records = UploadHistory.query.filter_by(user_id=user_id)\
                                             .order_by(UploadHistory.uploaded_at.desc())\
                                             .all()
        history_list = [record.to_dict() for record in history_records]
        return jsonify(history_list), 200
    except Exception as e:
        print(f"Error fetching history for user {user_id}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": "Failed to fetch history"}), 500
# ============================================


if __name__ == '__main__':
    print("--- Ensure migrations are up to date ---")
    print("Run: flask db migrate -m \"<message>\" and flask db upgrade")
    print(f"Flask static folder: {app.static_folder}")
    print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)