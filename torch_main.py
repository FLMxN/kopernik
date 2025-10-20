from safetensors.torch import load_file
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import torch
from torch import nn
import datasets
from datasets import Dataset, load_dataset
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import transforms, models
from umap import UMAP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import os
from launcher import predict
import pandas as pd
import plotly.express as px
from tqdm import tqdm

IMG = "pics/giper2.jpg"

FOV = 75
HORIZONTAL = 0
VERTICAL = 0
HEIGHT = 561
WIDTH = 997

sys.setrecursionlimit(10000)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

dataset = load_dataset("stochastic/random_streetview_images_pano_v0.0.2")
dataset = dataset.cast_column("image", datasets.Image())

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, num_classes=56):
        super().__init__()
        backbone = models.resnet50(weights=None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, num_classes)
        self.backbone = backbone

    def forward(self, x, return_features=False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        pooled = self.backbone.avgpool(x)
        feats = torch.flatten(pooled, 1)
        logits = self.backbone.fc(feats)

        return feats if return_features else logits

model = ResNet50FeatureExtractor(num_classes=56)
state_dict = torch.load('D:/resnet50-finetuned_raw/resnet50_streetview.pth', map_location='cuda')
model.load_state_dict(state_dict)
model.eval()

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained("D:/resnet50-finetuned")
    print("feature extractor found")
except:
    feature_extractor = None
    print("no feature extractor found, using manual preprocessing")

embeddings = []
labels = []

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

label2id_map = {
    "AD": 7, "AE": 16, "AR": 15, "AU": 43, "BD": 30,
    "BE": 26, "BG": 4, "BR": 46, "BT": 48, "BW": 31,
    "CA": 3, "CH": 49, "CL": 34, "CO": 17, "CZ": 45,
    "DE": 25, "DK": 36, "EE": 12, "ES": 41, "FI": 23,
    "FR": 28, "GB": 0, "GR": 53, "HK": 22, "HR": 24,
    "HU": 14, "ID": 42, "IE": 13, "IL": 51, "IS": 6,
    "IT": 27, "JP": 35, "KH": 10, "KR": 1, "LT": 32,
    "LV": 50, "MX": 29, "MY": 9, "NL": 2, "NO": 5,
    "NZ": 37, "PE": 44, "PL": 47, "PT": 21, "RO": 38,
    "RU": 52, "SE": 40, "SG": 19, "SI": 55, "SK": 8,
    "SZ": 11, "TH": 18, "TW": 33, "UA": 39, "US": 54,
    "ZA": 20
}

id2label_map = {
    7: "AD", 16: "AE", 15: "AR", 43: "AU", 30: "BD",
    26: "BE", 4: "BG", 46: "BR", 48: "BT", 31: "BW",
    3: "CA", 49: "CH", 34: "CL", 17: "CO", 45: "CZ",
    25: "DE", 36: "DK", 12: "EE", 41: "ES", 23: "FI",
    28: "FR", 0: "GB", 53: "GR", 22: "HK", 24: "HR",
    14: "HU", 42: "ID", 13: "IE", 51: "IL", 6: "IS",
    27: "IT", 35: "JP", 10: "KH", 1: "KR", 32: "LT",
    50: "LV", 29: "MX", 9: "MY", 2: "NL", 5: "NO",
    37: "NZ", 44: "PE", 47: "PL", 21: "PT", 38: "RO",
    52: "RU", 40: "SE", 19: "SG", 55: "SI", 8: "SK",
    11: "SZ", 18: "TH", 33: "TW", 39: "UA", 54: "US",
    20: "ZA"
}

iso_alpha2_to_country = {
    "AD": "Andorra",
    "AE": "United Arab Emirates",
    "AF": "Afghanistan",
    "AG": "Antigua and Barbuda",
    "AI": "Anguilla",
    "AL": "Albania",
    "AM": "Armenia",
    "AO": "Angola",
    "AQ": "Antarctica",
    "AR": "Argentina",
    "AS": "American Samoa",
    "AT": "Austria",
    "AU": "Australia",
    "AW": "Aruba",
    "AX": "Åland Islands",
    "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina",
    "BB": "Barbados",
    "BD": "Bangladesh",
    "BE": "Belgium",
    "BF": "Burkina Faso",
    "BG": "Bulgaria",
    "BH": "Bahrain",
    "BI": "Burundi",
    "BJ": "Benin",
    "BL": "Saint Barthélemy",
    "BM": "Bermuda",
    "BN": "Brunei Darussalam",
    "BO": "Bolivia (Plurinational State of)",
    "BQ": "Caribbean Netherlands",
    "BR": "Brazil",
    "BS": "The Bahamas",
    "BT": "Bhutan",
    "BV": "Bouvet Island",
    "BW": "Botswana",
    "BY": "Belarus",
    "BZ": "Belize",
    "CA": "Canada",
    "CC": "Cocos (Keeling) Islands",
    "CD": "Democratic Republic of the Congo",
    "CF": "Central African Republic",
    "CG": "Republic of the Congo",
    "CH": "Switzerland",
    "CI": "Ivory Coast",
    "CK": "Cook Islands",
    "CL": "Chile",
    "CM": "Cameroon",
    "CN": "People's Republic of China",
    "CO": "Colombia",
    "CR": "Costa Rica",
    "CU": "Cuba",
    "CV": "Cape Verde",
    "CW": "Curaçao",
    "CX": "Christmas Island",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DJ": "Djibouti",
    "DK": "Denmark",
    "DM": "Dominica",
    "DO": "Dominican Republic",
    "DZ": "Algeria",
    "EC": "Ecuador",
    "EE": "Estonia",
    "EG": "Egypt",
    "EH": "Western Sahara",
    "ER": "Eritrea",
    "ES": "Spain",
    "ET": "Ethiopia",
    "FI": "Finland",
    "FJ": "Fiji",
    "FM": "Federated States of Micronesia",
    "FO": "Faroe Islands",
    "FR": "France",
    "GA": "Gabon",
    "GB": "United Kingdom",
    "GD": "Grenada",
    "GE": "Georgia",
    "GF": "French Guiana",
    "GG": "Guernsey",
    "GH": "Ghana",
    "GI": "Gibraltar",
    "GL": "Greenland",
    "GM": "The Gambia",
    "GN": "Guinea",
    "GP": "Guadeloupe",
    "GQ": "Equatorial Guinea",
    "GR": "Greece",
    "GS": "South Georgia and the South Sandwich Islands",
    "GT": "Guatemala",
    "GU": "Guam",
    "GW": "Guinea-Bissau",
    "GY": "Guyana",
    "HK": "Hong Kong",
    "HM": "Heard Island and McDonald Islands",
    "HN": "Honduras",
    "HR": "Croatia",
    "HT": "Haiti",
    "HU": "Hungary",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IL": "Israel",
    "IM": "Isle of Man",
    "IN": "India",
    "IO": "British Indian Ocean Territory",
    "IQ": "Iraq",
    "IR": "Islamic Republic of Iran",
    "IS": "Iceland",
    "IT": "Italy",
    "JE": "Jersey",
    "JM": "Jamaica",
    "JO": "Jordan",
    "JP": "Japan",
    "KE": "Kenya",
    "KG": "Kyrgyzstan",
    "KH": "Cambodia",
    "KI": "Kiribati",
    "KM": "Comoros",
    "KN": "Saint Kitts and Nevis",
    "KP": "Democratic People's Republic of Korea",
    "KR": "Republic of Korea",
    "KW": "Kuwait",
    "KY": "Cayman Islands",
    "KZ": "Kazakhstan",
    "LA": "Lao People's Democratic Republic",
    "LB": "Lebanon",
    "LC": "Saint Lucia",
    "LI": "Liechtenstein",
    "LK": "Sri Lanka",
    "LR": "Liberia",
    "LS": "Lesotho",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "LY": "Libya",
    "MA": "Morocco",
    "MC": "Monaco",
    "MD": "Moldova (Republic of)",
    "ME": "Montenegro",
    "MF": "Saint Martin (French part)",
    "MG": "Madagascar",
    "MH": "Marshall Islands",
    "MK": "North Macedonia",
    "ML": "Mali",
    "MM": "Myanmar",
    "MN": "Mongolia",
    "MO": "Macao",
    "MP": "Northern Mariana Islands",
    "MQ": "Martinique",
    "MR": "Mauritania",
    "MS": "Montserrat",
    "MT": "Malta",
    "MU": "Mauritius",
    "MV": "Maldives",
    "MW": "Malawi",
    "MX": "Mexico",
    "MY": "Malaysia",
    "MZ": "Mozambique",
    "NA": "Namibia",
    "NC": "New Caledonia",
    "NE": "Niger",
    "NF": "Norfolk Island",
    "NG": "Nigeria",
    "NI": "Nicaragua",
    "NL": "Netherlands",
    "NO": "Norway",
    "NP": "Nepal",
    "NR": "Nauru",
    "NU": "Niue",
    "NZ": "New Zealand",
    "OM": "Oman",
    "PA": "Panama",
    "PE": "Peru",
    "PF": "French Polynesia",
    "PG": "Papua New Guinea",
    "PH": "Philippines",
    "PK": "Pakistan",
    "PL": "Poland",
    "PM": "Saint Pierre and Miquelon",
    "PN": "Pitcairn",
    "PR": "Puerto Rico",
    "PT": "Portugal",
    "PW": "Palau",
    "PY": "Paraguay",
    "QA": "Qatar",
    "RE": "Réunion",
    "RO": "Romania",
    "RS": "Serbia",
    "RU": "Russian Federation",
    "RW": "Rwanda",
    "SA": "Saudi Arabia",
    "SB": "Solomon Islands",
    "SC": "Seychelles",
    "SD": "Sudan",
    "SE": "Sweden",
    "SG": "Singapore",
    "SH": "Saint Helena, Ascension and Tristan da Cunha",
    "SI": "Slovenia",
    "SJ": "Svalbard and Jan Mayen",
    "SK": "Slovakia",
    "SL": "Sierra Leone",
    "SM": "San Marino",
    "SN": "Senegal",
    "SO": "Somalia",
    "SR": "Suriname",
    "SS": "South Sudan",
    "ST": "Sao Tome and Principe",
    "SV": "El Salvador",
    "SX": "Sint Maarten (Dutch part)",
    "SY": "Syrian Arab Republic",
    "SZ": "Eswatini",
    "TC": "Turks and Caicos Islands",
    "TD": "Chad",
    "TF": "French Southern and Antarctic Lands",
    "TG": "Togo",
    "TH": "Thailand",
    "TJ": "Tajikistan",
    "TK": "Tokelau",
    "TL": "Timor-Leste",
    "TM": "Turkmenistan",
    "TN": "Tunisia",
    "TO": "Tonga",
    "TR": "Turkey",
    "TT": "Trinidad and Tobago",
    "TV": "Tuvalu",
    "TZ": "Tanzania (United Republic of)",
    "TW": "Taiwan",
    "UA": "Ukraine",
    "UG": "Uganda",
    "UM": "United States Minor Outlying Islands",
    "US": "United States",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "VA": "Holy See",
    "VC": "Saint Vincent and the Grenadines",
    "VE": "Venezuela (Bolivarian Republic of)",
    "VG": "British Virgin Islands",
    "VI": "United States Virgin Islands",
    "VN": "Viet Nam",
    "VU": "Vanuatu",
    "WF": "Wallis and Futuna",
    "WS": "Samoa",
    "YE": "Yemen",
    "YT": "Mayotte",
    "ZA": "South Africa",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

def lowres(img, target_height=561):
    orig_width, orig_height = img.size
    aspect_ratio = orig_width / orig_height
    new_width = int(target_height * aspect_ratio)
    resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    return resized_img

# --- Embedding extraction (pure Torch) ---
def get_embeddings(model, dataloader, device):
    model.to(device)
    all_embs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            feats = model(pixel_values, return_features=True)
            all_embs.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_embs).numpy(), torch.cat(all_labels).numpy()

# --- Single image inference ---
def extract_sample_embedding(model, image_path, device):
    sample_raw = lowres(Image.open(image_path).convert('RGB'))
    sample_tensor = preprocess(sample_raw).unsqueeze(0).to(device)
    with torch.no_grad():
        sample_emb = model(sample_tensor, return_features=True).cpu().numpy()
    return sample_emb, sample_raw

# --- UMAP + visualization ---
def project_and_plot(centroids, sample_emb, classes, id2label_map, iso_alpha2_to_country):
    all_points = np.concatenate([centroids, sample_emb], axis=0)
    scaled = StandardScaler().fit_transform(all_points)
    umap_2d = UMAP(n_components=2, random_state=42).fit_transform(scaled)
    plane_centroids = umap_2d[:-1]
    plane_sample = umap_2d[-1]
    
    plt.figure(figsize=(16, 8))
    plt.scatter(plane_centroids[:, 0], plane_centroids[:, 1], c=classes, cmap='tab20', s=5)
    for i, lbl in enumerate(classes):
        country = id2label_map.get(int(lbl), str(lbl))
        plt.text(plane_centroids[i, 0], plane_centroids[i, 1], country, fontsize=10, ha='center', va='center')
    plt.scatter(plane_sample[0], plane_sample[1], c='red', s=80, edgecolor='black', marker='X')
    plt.show()

# --- Example run for a single image ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_emb, sample_raw = extract_sample_embedding(model, IMG, device)
    print("Extracted sample embedding:", sample_emb.shape)