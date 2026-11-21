from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from .paths import OUTPUT_DIR, ROOT_DIR
from .torch_gradcam import gradcam, extract_gradcam_blobs
import dotenv

dotenv_file = ROOT_DIR / '.env'
dotenv.load_dotenv(dotenv_file, override=True)
env = os.environ.copy()

# -------------------------------------------------------- CONFIG ----------------------------------------------------------
HEIGHT = 561
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

iso_alpha2_to_country = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AR": "Argentina", "AU": "Australia",
    "BD": "Bangladesh", "BE": "Belgium", "BG": "Bulgaria", "BR": "Brazil", "BT": "Bhutan",
    "BW": "Botswana", "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CO": "Colombia",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GR": "Greece", "HK": "Hong Kong",
    "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel",
    "IS": "Iceland", "IT": "Italy", "JP": "Japan", "KH": "Cambodia", "KR": "Republic of Korea",
    "LT": "Lithuania", "LV": "Latvia", "MX": "Mexico", "MY": "Malaysia", "NL": "Netherlands",
    "NO": "Norway", "NZ": "New Zealand", "PE": "Peru", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RU": "Russian Federation", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SZ": "Eswatini", "TH": "Thailand", "TW": "Taiwan",
    "UA": "Ukraine", "US": "United States", "ZA": "South Africa", "UNDEFINED": "Undefined"
}

reg_to_name = {
    "east_asia": "Indochine and East Asia", 
    "nordic": "Scandinavia and Northern Europe",
    "post_socialist": "Balkans and Eastern Europe",
    "anglosphere": "Anglosphere and Central Europe", 
    "mediterranean": "Mediterranean and South Europe",
    "tropical": "Latin America and South Asia",
    "arid_african": "Resorts and Undefined Nature"
}

def draw(model, data_dict, image_path, task, show_pictures, colormap, id_map, is_region, alpha):
    class_idx = []
    label2id = {i: l for l, i in id_map.items()}
    if is_region:
        for i in data_dict:
            key = list(data_dict.keys())[0]
            for label, name in reg_to_name.items():
                if name == key:
                    iso_code = label
            class_idx.append(label2id[iso_code])
    else:
        for i in data_dict:
            key = list(data_dict.keys())[0]
            iso_code = key.split('(')[-1].rstrip(')')
            class_idx.append(label2id[iso_code])
    
    img, gcam = gradcam(model, image_path, alpha=alpha, class_idx=class_idx, device=DEVICE, colormap=colormap, task=('region' if is_region else 'country'))
    blobs = extract_gradcam_blobs(gcam, img_size=(Image.open(image_path).size), threshold_percentile=85)
    # img = Image.open(f"pics/{image_path.split('/')[1]}")
    draw = ImageDraw.Draw(img)
    for blob in blobs:
        x1, y1, x2, y2 = blob['bbox']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    try:
        font_paths = [
            "/System/Library/Fonts/Menlo.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/consola.ttf"  # Windows
        ]
        
        font = None
        font_size = img.width // 64
        line_height = img.height // 32
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    x, y = img.width - font_size*24, img.height - (line_height * (len(data_dict) + 1))
    match task:
        case "region":
            fill = (255, 255, 0)
        case "country":
            fill = (0, 255, 255)

    for key, value in sorted(data_dict.items(), key=lambda item: item[1], reverse=True):
        line = f"{key}: {value}"
        draw.text((x, y), line, fill=fill, font=font)       
        y += line_height

    output_filename = f'{task}_{os.path.basename(image_path)}'
    if show_pictures:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        img.save(OUTPUT_DIR / output_filename)
        try:
            dotenv.set_key(dotenv_file, f"{task.upper()}_IMG", f"output/{output_filename}")
        except Exception:
            pass
        return f'/output/{output_filename}'

    return None

    # img.save(f"output/{image_path.split('/')[1]}")
    # if show_pictures:
    #     try:
    #         img.show()
    #     except Exception as e:
    #         print(f"Skipping display. Unable to show image: {e}")


def predict_country(model, samples, show_pictures, top_k=5, device=DEVICE, IS_PRETTY=False):
    country_out = None
    if IS_PRETTY:
        state_score = {
        "AD":0, "AE":0, "AR":0, "AU":0,
        "BD":0, "BE":0, "BG":0, "BR":0, "BT":0,
        "BW":0, "CA":0, "CH":0, "CL":0, "CO":0,
        "CZ":0, "DE":0, "DK":0, "EE":0, "ES":0,
        "FI":0, "FR":0, "GB":0, "GR":0, "HK":0,
        "HR":0, "HU":0, "ID":0, "IE":0, "IL":0,
        "IS":0, "IT":0, "JP": 0, "KH":0, "KR":0,
        "LT":0, "LV":0, "MX":0, "MY":0, "NL":0,
        "NO":0, "NZ":0, "PE":0, "PL":0, "PT":0,
        "RO":0, "RU":0, "SE":0, "SG":0,
        "SI":0, "SK":0, "SZ":0, "TH":0, "TW":0,
        "UA":0, "US":0, "ZA":0, "UNDEFINED": 0
    }
        regions = {
            "Europe": ["PL", "AD", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LV", "NL", "NO", "PT", "RO", "RU", "SE", "SI", "SK", "UA"],
            "Asia": ["AE", "BD", "HK", "ID", "IL", "JP", "KR", "MY", "SG", "TH", "TW"],
            "Oceania": ["AU"],
            "North America": ["CA", "MX", "US"],
            "South America": ["BR", "CL", "CO", "PE"],
            "Africa": ["BW", "ZA"],
            "Undefined": ["MY", "SI", "CH", "CA", "DE"]
        }
        unk_score = 0
    else:
        regions = {
            "Europe": ["AD", "PL", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LV", "NL", "NO", "PT", "RO", "RU", "SE", "SI", "SK", "UA"],
            "Asia": ["AE", "BT", "BD", "KH", "HK", "ID", "IL", "JP", "KR", "MY", "SG", "TH", "TW"],
            "Oceania": ["AU", "NZ"],
            "North America": ["CA", "MX", "US"],
            "South America": ["BR", "AR", "CL", "CO", "PE"],
            "Africa": ["BW", "SZ", "ZA"]
        }
        state_score = {
        "AD":0, "AE":0, "AR":0, "AU":0,
        "BD":0, "BE":0, "BG":0, "BR":0, "BT":0,
        "BW":0, "CA":0, "CH":0, "CL":0, "CO":0,
        "CZ":0, "DE":0, "DK":0, "EE":0, "ES":0,
        "FI":0, "FR":0, "GB":0, "GR":0, "HK":0,
        "HR":0, "HU":0, "ID":0, "IE":0, "IL":0,
        "IS":0, "IT":0, "JP": 0, "KH":0, "KR":0,
        "LT":0, "LV":0, "MX":0, "MY":0, "NL":0,
        "NO":0, "NZ":0, "PE":0, "PL":0, "PT":0,
        "RO":0, "RU":0, "SE":0, "SG":0,
        "SI":0, "SK":0, "SZ":0, "TH":0, "TW":0,
        "UA":0, "US":0, "ZA":0
        }
    
    eu_score = 0
    asia_score = 0
    ocean_score = 0
    na_score = 0
    sa_score = 0
    africa_score = 0

    longitudes = []
    latitudes = []

    for x in enumerate(samples, 0):
        if x[0]%2 == 0:
            draw_buffer = {}
        resized_img = samples[x[0]][0]
        inputs = preprocess(resized_img).unsqueeze(0)
        inputs = inputs.to(device)

        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except:
                outputs = model(inputs)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('classification_logits'))
            longitude = outputs['longitude'].cpu().numpy()[0]
            latitude = outputs['latitude'].cpu().numpy()[0]
        elif isinstance(outputs, tuple):
            logits = outputs[0]
            if len(outputs) > 1:
                # Assuming outputs[1] contains coordinates
                coords = outputs[1].cpu().numpy()[0]
                if len(coords) == 2:
                    longitude, latitude = coords[0], coords[1]
        else:
            logits = outputs
        
        if 'longitude' in locals() and 'latitude' in locals():
            if not IS_PRETTY:
                print(f"\n{x} is loaded")
                # print(f"\nCoordinates of {x}: {longitude}, {latitude}")
            longitudes.append(longitude)
            latitudes.append(latitude)
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        if not IS_PRETTY:
            reg_top_probs, reg_top_indices = torch.topk(probabilities, 56)
            reg_top_probs = reg_top_probs.cpu().numpy()[0]
            reg_top_indices = reg_top_indices.cpu().numpy()[0]

        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = model.id2label[idx]             
            state_score[label] = state_score[label] + prob*100/len(samples)
            draw_key = f"{iso_alpha2_to_country[label]} ({label})"
            if x[0]%2 == 0 or x[0] == 0:
                draw_buffer[draw_key] = round(float(prob*100/2), ndigits=2)
            else:
                draw_buffer[draw_key] = round(float(draw_buffer.get(draw_key, 0.0) + prob*100/2), ndigits=2)

        if x[0]%2 != 0:
            country_out = draw(model=model, data_dict=draw_buffer, image_path=samples[x[0]][1], task="country", show_pictures=show_pictures, colormap='turbo', id_map=model.id2label, is_region=False, alpha=0.5)

        if not IS_PRETTY:
            for i, (prob, idx) in enumerate(zip(reg_top_probs, reg_top_indices)):
                label = model.id2label[idx]

                match label:
                    case country if country in regions["Europe"]:
                        eu_score = eu_score + prob
                    case country if country in regions["Asia"]:
                        asia_score = asia_score + prob
                    case country if country in regions["North America"]:
                        na_score = na_score + prob
                    case country if country in regions["South America"]:
                        sa_score = sa_score + prob
                    case country if country in regions["Oceania"]:
                        ocean_score = ocean_score + prob
                    case country if country in regions["Africa"]:
                        africa_score = africa_score + prob
                    case country if country in regions["Undefined"]:
                        unk_score = unk_score + prob
    
    if IS_PRETTY:
        for i, n in state_score.items():
            if i in regions["Undefined"]:
                state_score["UNDEFINED"] += n
                state_score[i] = 0

    preds = dict(sorted(
    ((f"{iso_alpha2_to_country[k]} ({k})", float(v)) for k, v in state_score.items() if v != 0),
    key=lambda x: x[1],
    reverse=True))

    # print(f"\nCoordinates: {mean(longitudes)}, {mean(latitudes)}") #fuck this shit

    if not IS_PRETTY:
        print(f"\nContinental predictions:")
        print(f"    Europe: {eu_score*100/len(samples):.2f}")
        print(f"    Asia: {asia_score*100/len(samples):.2f}")
        print(f"    North America: {na_score*100/len(samples):.2f}")
        print(f"    South America: {sa_score*100/len(samples):.2f}")
        print(f"    Oceania: {ocean_score*100/len(samples):.2f}")
        print(f"    Africa: {africa_score*100/len(samples):.2f}")

    print(f"\nParticular predictions:")
    for y in preds:
        print(f"    {y}: {preds[y]:.2f}")

    return {
        "predictions": {k: round(v, 2) for k, v in preds.items()},
        "gradcam": country_out
    }

def predict_region(model, samples, show_pictures, top_k=3, device=DEVICE, IS_PRETTY=False):
    region_out = None
    regions = {"east_asia":0, "nordic":0, "post_socialist":0, "anglosphere":0, "mediterranean":0, "tropical":0, "arid_african":0}
    for x in enumerate(samples, 0):
        if x[0]%2 == 0:
            draw_buffer = {}
        resized_img = samples[x[0]][0]
        inputs = preprocess(resized_img).unsqueeze(0)
        inputs = inputs.to(device)

        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except:
                outputs = model(inputs)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('classification_logits'))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        if not IS_PRETTY:
            print(f"\n{x} is loaded")
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = model.id2label[idx]
            regions[label] = regions[label] + prob*100/len(samples)

            draw_key = reg_to_name[label]
            if x[0]%2 == 0 or x[0] == 0:
                draw_buffer[draw_key] = round(float(prob*100/2), ndigits=2)
            else:
                draw_buffer[draw_key] = round(float(draw_buffer.get(draw_key, 0.0) + prob*100/2), ndigits=2)
        if x[0]%2 != 0:
            region_out = draw(model=model, data_dict=draw_buffer, image_path=samples[x[0]][1], task="region", show_pictures=show_pictures, colormap='plasma', id_map=model.id2label, is_region=True, alpha=0.5)             

        preds = dict(sorted(
    ((reg_to_name[k], float(v)) for k, v in regions.items() if v != 0),
    key=lambda x: x[1],
    reverse=True))
        
    print(f"\nRegional predictions:")
    for y in preds:
        print(f"    {y}: {preds[y]:.2f}")

    return {
        "predictions": {k: round(v, 2) for k, v in preds.items()},
        "gradcam": region_out
    }
