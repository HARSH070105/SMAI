"""
Scrapes and appends metadata for the 26 NEW monument classes
(beyond the original 24) to metadata/monuments_metadata.json.

Run AFTER scrape_metadata.py has already created the base JSON.
Usage: python scripts/scrape_new_metadata.py
"""

import os
import json
import time
import requests
from urllib.parse import quote

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_API  = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
HEADERS   = {"User-Agent": "SMAI-A3-MonumentApp/1.0 (academic project)"}

# ==========================================
# Static data for 26 new monuments
# Keys match folder names in add_new_classes.py
# ==========================================

NEW_MONUMENT_STATIC = {
    # T12.2 — Forts of Maharashtra
    "Raigad Fort": {
        "display_name": "Raigad Fort",
        "wikipedia_query": "Raigad Fort",
        "location": "Mahad, Raigad District, Maharashtra",
        "state": "Maharashtra",
        "lat": 18.2346, "lon": 73.4410,
        "opening_hours": "8:00 AM – 5:30 PM",
        "ticket_price": "₹20 (Indians) | ₹250 (Foreigners)",
        "category": "Hill Fort",
        "fun_facts": [
            "Chhatrapati Shivaji Maharaj was crowned here in 1674, establishing the Maratha Empire.",
            "Stands 820 metres above sea level; accessible by a ropeway or a 1,400-step climb.",
            "Shivaji's samadhi (memorial) and the Jagdishwar temple are located within the fort.",
            "Once called 'Rairi', it was renamed Raigad (King's Fort) after Shivaji captured it in 1656.",
        ],
    },
    "Sinhagad Fort": {
        "display_name": "Sinhagad Fort",
        "wikipedia_query": "Sinhagad",
        "location": "Pune District, Maharashtra",
        "state": "Maharashtra",
        "lat": 18.3659, "lon": 73.7554,
        "opening_hours": "6:00 AM – 6:00 PM",
        "ticket_price": "₹25 (Indians)",
        "category": "Hill Fort",
        "fun_facts": [
            "Site of the famous 1670 Battle of Sinhagad where Tanaji Malusare died recapturing it for Shivaji.",
            "Shivaji reportedly said 'Gad ala pan Sinha gela' (the fort was won but the lion was lost) upon Tanaji's death.",
            "Stands 1,312 metres above sea level in the Sahyadri mountain range.",
            "Originally called Kondana; renamed Sinhagad (Lion's Fort) after the heroic battle.",
        ],
    },
    "Pratapgad Fort": {
        "display_name": "Pratapgad Fort",
        "wikipedia_query": "Pratapgad",
        "location": "Satara District, Maharashtra",
        "state": "Maharashtra",
        "lat": 17.9376, "lon": 73.5793,
        "opening_hours": "9:00 AM – 5:00 PM",
        "ticket_price": "₹20 (Indians)",
        "category": "Hill Fort",
        "fun_facts": [
            "Site of the 1659 Battle of Pratapgad where Shivaji killed Afzal Khan of the Bijapur Sultanate.",
            "Built in 1656 by Shivaji Maharaj under the supervision of his general Moropant Trimbak Pingle.",
            "Houses the temple of Bhavani Mata, the tutelary deity of the Maratha Empire.",
            "Stands at 1,080 metres; the fort has upper and lower sections connected by stone pathways.",
        ],
    },
    "Shivneri Fort": {
        "display_name": "Shivneri Fort",
        "wikipedia_query": "Shivneri",
        "location": "Junnar, Pune District, Maharashtra",
        "state": "Maharashtra",
        "lat": 19.2027, "lon": 73.8728,
        "opening_hours": "8:00 AM – 6:00 PM",
        "ticket_price": "Free",
        "category": "Hill Fort",
        "fun_facts": [
            "Birthplace of Chhatrapati Shivaji Maharaj, born here on 19 February 1630.",
            "Contains a statue of Shivaji as an infant with his mother Jijabai inside the fort.",
            "Features a temple dedicated to the goddess Shivai, after whom Shivaji is named.",
            "Houses ancient Buddhist cave temples dating back to the 1st–2nd century CE.",
        ],
    },
    "Lohagad Fort": {
        "display_name": "Lohagad Fort",
        "wikipedia_query": "Lohagad",
        "location": "Maval, Pune District, Maharashtra",
        "state": "Maharashtra",
        "lat": 18.7456, "lon": 73.4757,
        "opening_hours": "6:00 AM – 6:00 PM",
        "ticket_price": "Free",
        "category": "Hill Fort",
        "fun_facts": [
            "Stands at 1,033 metres above sea level in the Sahyadri range near Lonavala.",
            "Shivaji used this fort to store war treasury after the Surat raids of 1664 and 1670.",
            "The 'Vinchukata' (Scorpion's Tail) is a distinctive winding wall leading to the main gate.",
            "One of the earliest forts in Maharashtra, with records dating back to the 1st century BCE.",
        ],
    },
    "Panhala Fort": {
        "display_name": "Panhala Fort",
        "wikipedia_query": "Panhala",
        "location": "Kolhapur District, Maharashtra",
        "state": "Maharashtra",
        "lat": 16.8130, "lon": 74.1120,
        "opening_hours": "8:00 AM – 6:00 PM",
        "ticket_price": "Free",
        "category": "Hill Fort",
        "fun_facts": [
            "The largest fort in the Deccan; stretches over 7 km with a circumference of 14 km.",
            "Shivaji escaped from Mughal siege here in 1660 in one of history's most daring escapes.",
            "Built in the 12th century by Raja Bhoja II; served as Shivaji's residence for 7 years.",
            "Three large grain storehouses within the fort could hold enough supplies for years of siege.",
        ],
    },
    "Sindhudurg Fort": {
        "display_name": "Sindhudurg Fort",
        "wikipedia_query": "Sindhudurg",
        "location": "Malvan, Sindhudurg District, Maharashtra",
        "state": "Maharashtra",
        "lat": 16.0333, "lon": 73.5083,
        "opening_hours": "8:00 AM – 6:00 PM",
        "ticket_price": "₹10 (Indians)",
        "category": "Sea Fort",
        "fun_facts": [
            "Built by Shivaji Maharaj between 1664 and 1667 on a rocky island in the Arabian Sea.",
            "Construction used lead and iron at the base to fix the foundation to the sea rocks.",
            "Houses a rare temple with impressions of Shivaji's hand, foot, and chest cast in lime.",
            "Took three years and employed 4,000 workers; it was India's first purpose-built naval fort.",
        ],
    },

    # T12.3 — Temples of Tamil Nadu
    "Meenakshi Temple": {
        "display_name": "Meenakshi Amman Temple",
        "wikipedia_query": "Meenakshi Amman Temple",
        "location": "Madurai, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 9.9195, "lon": 78.1193,
        "opening_hours": "5:00 AM – 12:30 PM and 4:00 PM – 10:00 PM",
        "ticket_price": "Free (camera fee applies)",
        "category": "Hindu Temple (Dravidian Architecture)",
        "fun_facts": [
            "Has 14 gopurams (gateway towers); the tallest is 52 metres and covered with 33,000 sculptures.",
            "Dedicated to Goddess Meenakshi (a form of Parvati) and her consort Sundareswarar (Shiva).",
            "Attracts 15,000–20,000 visitors daily; up to 25,000 on Fridays.",
            "The temple complex covers 6 hectares and is considered the cultural heart of Madurai.",
        ],
    },
    "Shore Temple": {
        "display_name": "Shore Temple",
        "wikipedia_query": "Shore Temple",
        "location": "Mahabalipuram, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 12.6175, "lon": 80.1993,
        "opening_hours": "6:00 AM – 6:00 PM",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Hindu Temple",
        "fun_facts": [
            "One of the oldest structural stone temples of South India, built in the 8th century CE by the Pallava dynasty.",
            "Faces east so that sunlight strikes the deity's face at dawn — a deliberate architectural feat.",
            "UNESCO World Heritage Site since 1984 as part of 'Group of Monuments at Mahabalipuram'.",
            "Local legend says six more temples lie submerged beneath the sea — some ruins were revealed after the 2004 tsunami.",
        ],
    },
    "Ramanathaswamy Temple": {
        "display_name": "Ramanathaswamy Temple",
        "wikipedia_query": "Ramanathaswamy Temple",
        "location": "Rameswaram, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 9.2885, "lon": 79.3174,
        "opening_hours": "5:00 AM – 1:00 PM and 3:00 PM – 9:00 PM",
        "ticket_price": "Free",
        "category": "Hindu Temple (Pilgrimage Site)",
        "fun_facts": [
            "Has the longest corridor of any Hindu temple in the world — the third corridor is 1,212 metres long.",
            "One of the Char Dham pilgrimage sites sacred to Hindus, along with Badrinath, Dwarka, and Puri.",
            "Houses 22 sacred wells (theerthams); pilgrims traditionally bathe in all of them.",
            "Legend says Rama built a Shiva lingam here before crossing to Lanka to rescue Sita.",
        ],
    },
    "Kapaleeshwarar Temple": {
        "display_name": "Kapaleeshwarar Temple",
        "wikipedia_query": "Kapaleeshwarar Temple",
        "location": "Mylapore, Chennai, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 13.0338, "lon": 80.2691,
        "opening_hours": "5:00 AM – 12:00 PM and 4:00 PM – 9:30 PM",
        "ticket_price": "Free",
        "category": "Hindu Temple (Dravidian Architecture)",
        "fun_facts": [
            "Dedicated to Shiva as Kapaleeshwarar and Parvati as Karpagambal.",
            "The present temple was rebuilt in the 16th century after the original was destroyed by the Portuguese.",
            "Features a stunning 37-metre tall gopuram (gateway tower) with colourful stucco sculptures.",
            "The annual Arubathimoovar festival drawing 1.5 lakh devotees is one of Chennai's biggest events.",
        ],
    },
    "Ranganathaswamy Temple": {
        "display_name": "Ranganathaswamy Temple (Srirangam)",
        "wikipedia_query": "Ranganathaswamy Temple, Srirangam",
        "location": "Srirangam, Tiruchirappalli, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 10.8619, "lon": 78.6892,
        "opening_hours": "6:00 AM – 1:00 PM and 3:15 PM – 9:00 PM",
        "ticket_price": "Free",
        "category": "Hindu Temple (Vaishnavite)",
        "fun_facts": [
            "The largest functioning Hindu temple in the world, covering 156 acres with 21 gopurams.",
            "Dedicated to Ranganatha, a reclining form of Vishnu — one of the 108 Divya Desams.",
            "The outermost wall of the complex is nearly 4 km long, enclosing a self-contained township.",
            "The Rajagopuram (main tower) stands 73 metres tall and took over 400 years to complete.",
        ],
    },
    "Nataraja Temple": {
        "display_name": "Thillai Nataraja Temple",
        "wikipedia_query": "Thillai Nataraja Temple, Chidambaram",
        "location": "Chidambaram, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 11.3994, "lon": 79.6934,
        "opening_hours": "6:00 AM – 12:00 PM and 5:00 PM – 10:00 PM",
        "ticket_price": "Free",
        "category": "Hindu Temple (Shaivite)",
        "fun_facts": [
            "One of the Pancha Bhuta Stalas (five elemental temples); represents the element of space (akasha).",
            "The dancing form of Shiva (Nataraja) here inspired the famous bronze idol now iconic worldwide.",
            "The temple roof is covered with 21,600 gold tiles representing the number of breaths per day.",
            "Over 2,000 years old; its current structure was built primarily by the Chola dynasty.",
        ],
    },

    # T12.4 — Mughal architecture (new ones)
    "Agra Fort": {
        "display_name": "Agra Fort",
        "wikipedia_query": "Agra Fort",
        "location": "Agra, Uttar Pradesh",
        "state": "Uttar Pradesh",
        "lat": 27.1795, "lon": 78.0211,
        "opening_hours": "6:00 AM – 6:00 PM",
        "ticket_price": "₹40 (Indians) | ₹550 (Foreigners)",
        "category": "Mughal Fort",
        "fun_facts": [
            "A UNESCO World Heritage Site; served as the main residence of the Mughal emperors until 1638.",
            "Shah Jahan was imprisoned here by his son Aurangzeb in his final years, with a view of the Taj Mahal.",
            "The fort contains over 500 buildings including palaces, mosques, and audience halls.",
            "Built primarily by Emperor Akbar starting in 1565 using red sandstone brought from Rajputana.",
        ],
    },
    "Red Fort Delhi": {
        "display_name": "Red Fort",
        "wikipedia_query": "Red Fort",
        "location": "Chandni Chowk, New Delhi",
        "state": "Delhi",
        "lat": 28.6562, "lon": 77.2410,
        "opening_hours": "9:30 AM – 4:30 PM (Closed on Mondays)",
        "ticket_price": "₹35 (Indians) | ₹500 (Foreigners)",
        "category": "Mughal Fort",
        "fun_facts": [
            "Built by Shah Jahan between 1638 and 1648; served as the Mughal capital for 200 years.",
            "India's Prime Minister hoists the national flag here every Independence Day (15 August).",
            "The fort was designed to be octagonal — its walls extend 2.41 km and stand up to 33 metres tall.",
            "UNESCO World Heritage Site since 2007.",
        ],
    },
    "Jama Masjid Delhi": {
        "display_name": "Jama Masjid",
        "wikipedia_query": "Jama Masjid, Delhi",
        "location": "Chandni Chowk, New Delhi",
        "state": "Delhi",
        "lat": 28.6507, "lon": 77.2334,
        "opening_hours": "7:00 AM – 12:00 PM and 1:30 PM – 6:30 PM",
        "ticket_price": "Free (camera fee: ₹300)",
        "category": "Mughal Mosque",
        "fun_facts": [
            "The largest mosque in India; can accommodate 25,000 worshippers at one time.",
            "Built by Shah Jahan between 1644 and 1656 using red sandstone and white marble.",
            "Has three gateways, two minarets, and a courtyard measuring 100×90 metres.",
            "Houses relics of the Prophet Muhammad including a hair, a sandal, and his footprint in marble.",
        ],
    },
    "Itmad-ud-Daulah": {
        "display_name": "Itmad-ud-Daulah (Baby Taj)",
        "wikipedia_query": "Itmad-ud-Daulah",
        "location": "Agra, Uttar Pradesh",
        "state": "Uttar Pradesh",
        "lat": 27.1922, "lon": 78.0386,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹30 (Indians) | ₹310 (Foreigners)",
        "category": "Mughal Tomb",
        "fun_facts": [
            "Often called the 'Baby Taj' — widely considered a forerunner to the Taj Mahal.",
            "First Mughal structure built entirely of white marble with pietra dura inlay work.",
            "Built by Nur Jahan for her father Mirza Ghiyas Beg, who served as treasurer to Emperor Jahangir.",
            "The intricate semi-precious stone mosaic work (parchin kari) here was the prototype used in the Taj Mahal.",
        ],
    },
    "Bibi Ka Maqbara": {
        "display_name": "Bibi Ka Maqbara",
        "wikipedia_query": "Bibi Ka Maqbara",
        "location": "Aurangabad, Maharashtra",
        "state": "Maharashtra",
        "lat": 19.9038, "lon": 75.3174,
        "opening_hours": "8:00 AM – 8:00 PM",
        "ticket_price": "₹25 (Indians) | ₹300 (Foreigners)",
        "category": "Mughal Tomb",
        "fun_facts": [
            "Built in 1660 by Prince Azam Shah for his mother Dilras Banu Begum, wife of Aurangzeb.",
            "So closely resembles the Taj Mahal that it is nicknamed the 'Taj of the Deccan.'",
            "Built on a limited budget; only the lower portion uses marble — the upper part is lime plaster.",
            "Stands in a large charbagh (four-part garden) similar to the Taj Mahal's layout.",
        ],
    },
    "Safdarjung Tomb": {
        "display_name": "Safdarjung's Tomb",
        "wikipedia_query": "Safdarjung's Tomb",
        "location": "New Delhi",
        "state": "Delhi",
        "lat": 28.5912, "lon": 77.2037,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹25 (Indians) | ₹310 (Foreigners)",
        "category": "Mughal Tomb",
        "fun_facts": [
            "Built in 1754 for Safdarjung, the last Prime Minister (wazir) of the Mughal Empire.",
            "Considered the last great Mughal garden tomb; its construction marked the end of the Mughal architectural tradition.",
            "Built using marble and red sandstone salvaged from the tomb of Abdul Rahim Khan-i-Khana.",
            "The main chamber houses Safdarjung's grave alongside those of his family members.",
        ],
    },
    "Akbar Tomb Sikandra": {
        "display_name": "Akbar's Tomb, Sikandra",
        "wikipedia_query": "Tomb of Akbar the Great",
        "location": "Sikandra, Agra, Uttar Pradesh",
        "state": "Uttar Pradesh",
        "lat": 27.2032, "lon": 77.9631,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹30 (Indians) | ₹310 (Foreigners)",
        "category": "Mughal Tomb",
        "fun_facts": [
            "Emperor Akbar began designing his own tomb during his lifetime — a unique Mughal tradition.",
            "Completed by his son Jahangir in 1613, who modified the original design significantly.",
            "The tomb is five storeys high; the top storey is open-air and made of white marble.",
            "The main gateway (Buland Darwaza) is decorated with white marble inlay and mosaics.",
        ],
    },

    # T12.5 — Hampi monuments
    "Virupaksha Temple Hampi": {
        "display_name": "Virupaksha Temple, Hampi",
        "wikipedia_query": "Virupaksha Temple, Hampi",
        "location": "Hampi, Karnataka",
        "state": "Karnataka",
        "lat": 15.3350, "lon": 76.4600,
        "opening_hours": "6:00 AM – 12:30 PM and 5:00 PM – 9:00 PM",
        "ticket_price": "Free",
        "category": "Hindu Temple (Vijayanagara Architecture)",
        "fun_facts": [
            "One of the oldest continuously functioning temples in India, active since at least the 7th century CE.",
            "The main gopuram (tower) stands 50 metres high with nine tiers.",
            "UNESCO World Heritage Site as part of the 'Group of Monuments at Hampi' since 1986.",
            "Dedicated to Shiva as Virupaksha, consort of the local goddess Pampa after whom the Tungabhadra River is named.",
        ],
    },
    "Vittala Temple Hampi": {
        "display_name": "Vittala Temple, Hampi",
        "wikipedia_query": "Vittala Temple, Hampi",
        "location": "Hampi, Karnataka",
        "state": "Karnataka",
        "lat": 15.3388, "lon": 76.4745,
        "opening_hours": "8:30 AM – 5:30 PM",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Hindu Temple (Vijayanagara Architecture)",
        "fun_facts": [
            "Famous for its 'musical pillars' that produce different musical notes when tapped.",
            "The stone chariot in the courtyard — with wheels that once actually rotated — is one of India's most iconic images.",
            "Built in the 15th–16th century during the Vijayanagara Empire's peak.",
            "UNESCO World Heritage Site as part of the 'Group of Monuments at Hampi' since 1986.",
        ],
    },
    "Lotus Mahal Hampi": {
        "display_name": "Lotus Mahal, Hampi",
        "wikipedia_query": "Lotus Mahal",
        "location": "Hampi, Karnataka",
        "state": "Karnataka",
        "lat": 15.3243, "lon": 76.4616,
        "opening_hours": "8:30 AM – 5:30 PM",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners) — part of Royal Enclosure",
        "category": "Vijayanagara Palace Pavilion",
        "fun_facts": [
            "A two-storeyed pavilion combining Hindu and Islamic architectural styles.",
            "Named for its arches that resemble lotus petals.",
            "Believed to have been a relaxation space for queens of the Vijayanagara Empire.",
            "UNESCO World Heritage Site as part of the 'Group of Monuments at Hampi' since 1986.",
        ],
    },

    # T12.6 — Stepwells of India
    "Rani ki Vav": {
        "display_name": "Rani ki Vav",
        "wikipedia_query": "Rani ki vav",
        "location": "Patan, Gujarat",
        "state": "Gujarat",
        "lat": 23.8588, "lon": 72.1013,
        "opening_hours": "8:00 AM – 6:00 PM",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Stepwell",
        "fun_facts": [
            "UNESCO World Heritage Site since 2014; featured on the Indian ₹100 note.",
            "Built in the 11th century by Queen Udayamati in memory of her husband King Bhimdev I.",
            "Has seven levels of stairs and over 500 principal sculptures and 1,000 minor ones.",
            "Was submerged in silt for centuries; excavated and restored by the Archaeological Survey of India in the 1980s.",
        ],
    },
    "Chand Baori": {
        "display_name": "Chand Baori",
        "wikipedia_query": "Chand Baori",
        "location": "Abhaneri, Rajasthan",
        "state": "Rajasthan",
        "lat": 27.0082, "lon": 76.6050,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "Free",
        "category": "Stepwell",
        "fun_facts": [
            "One of the largest and deepest stepwells in the world — 13 storeys deep with 3,500 steps.",
            "Built in the 9th century CE by King Chanda of the Nikumbha dynasty.",
            "The geometric precision of its steps creates stunning mirror-image patterns.",
            "Featured in films like 'The Dark Knight Rises' and 'The Fall'.",
        ],
    },
    "Adalaj Stepwell": {
        "display_name": "Adalaj Stepwell",
        "wikipedia_query": "Adalaj stepwell",
        "location": "Adalaj, Gandhinagar District, Gujarat",
        "state": "Gujarat",
        "lat": 23.1674, "lon": 72.5800,
        "opening_hours": "8:00 AM – 6:00 PM",
        "ticket_price": "Free",
        "category": "Stepwell",
        "fun_facts": [
            "Built in 1499 by Queen Rudabai in memory of her husband, Veer Singh, a Vaghela chieftain.",
            "Goes five storeys underground; the design keeps the interior cool even in summer.",
            "Combines Hindu and Islamic decorative motifs — a rare example of architectural syncretism.",
            "Has three entrance staircases that merge at the first floor into a single large platform.",
        ],
    },
}


# ==========================================
# Wikipedia fetcher
# ==========================================

def fetch_wiki_summary(query, sentences=4):
    title = quote(query.replace(" ", "_"), safe="()'")
    url = WIKI_API.format(title)
    for _ in range(2):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                time.sleep(1)
                continue
            data = resp.json()
            if data.get("type") == "disambiguation":
                return None
            extract = data.get("extract", "")
            if not extract:
                time.sleep(1)
                continue
            parts = extract.split(". ")
            return ". ".join(parts[:sentences]).strip() + "."
        except Exception:
            time.sleep(1)
    return None


# ==========================================
# Main — append to existing JSON
# ==========================================

output_path = os.path.join(ROOT_DIR, "metadata", "monuments_metadata.json")

# Load existing metadata
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded existing metadata: {len(metadata)} monuments")
else:
    metadata = {}
    print("No existing metadata found — creating fresh.")

print(f"Adding {len(NEW_MONUMENT_STATIC)} new monuments...\n")
failed_wiki = []

for i, (key, static) in enumerate(NEW_MONUMENT_STATIC.items(), 1):
    display = static["display_name"]
    print(f"  [{i:02d}/{len(NEW_MONUMENT_STATIC)}] {display}", end=" ... ", flush=True)

    history = fetch_wiki_summary(static["wikipedia_query"])
    time.sleep(0.3)

    if history is None:
        history = static.get("history") or " ".join(static["fun_facts"])
        failed_wiki.append(display)
        print("fallback")
    else:
        print("ok")

    google_maps_url = f"https://maps.google.com/?q={static['lat']},{static['lon']}"
    wiki_title      = quote(static["wikipedia_query"].replace(" ", "_"), safe="()'")
    wikipedia_url   = f"https://en.wikipedia.org/wiki/{wiki_title}"

    metadata[key] = {
        "display_name":  display,
        "location":      static["location"],
        "state":         static["state"],
        "lat":           static["lat"],
        "lon":           static["lon"],
        "opening_hours": static["opening_hours"],
        "ticket_price":  static["ticket_price"],
        "category":      static["category"],
        "history":       history,
        "fun_facts":     static["fun_facts"],
        "google_maps":   google_maps_url,
        "wikipedia_url": wikipedia_url,
    }

# Save updated JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(metadata)} total monuments written to:")
print(f"  {output_path}")

if failed_wiki:
    print(f"\nWikipedia fallback used for {len(failed_wiki)} monument(s):")
    for name in failed_wiki:
        print(f"  - {name}")
