"""
Scrapes and assembles metadata for all 24 monuments.
Structured data (location, hours, tickets, fun facts) is hardcoded for reliability.
History summaries are fetched from Wikipedia; falls back to hardcoded text on failure.
Output: metadata/monuments_metadata.json
"""

import os
import json
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. Static data for all 24 monuments
#    Keys match the class/folder names in monuments_24
# ==========================================

MONUMENT_STATIC = {
    "Ajanta Caves": {
        "display_name": "Ajanta Caves",
        "wikipedia_query": "Ajanta Caves",
        "location": "Aurangabad, Maharashtra",
        "state": "Maharashtra",
        "lat": 20.5519,
        "lon": 75.7033,
        "opening_hours": "9:00 AM – 5:30 PM (Closed on Mondays)",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Buddhist Cave Monuments",
        "fun_facts": [
            "Contains 30 rock-cut Buddhist caves dating back to the 2nd century BCE.",
            "Forgotten for over 1,000 years; rediscovered by British officer John Smith in 1819.",
            "The paintings inside use natural pigments and have survived for nearly 2,000 years.",
            "UNESCO World Heritage Site since 1983.",
        ],
    },
    "Charar-E- Sharif": {
        "display_name": "Charar-E-Sharif",
        "wikipedia_query": "Charar-e-Sharif",
        "location": "Budgam, Jammu & Kashmir",
        "state": "Jammu & Kashmir",
        "lat": 33.8662,
        "lon": 74.7951,
        "opening_hours": "Open daily (Dawn to Dusk)",
        "ticket_price": "Free",
        "category": "Sufi Shrine",
        "fun_facts": [
            "Shrine of Sheikh Noor-ud-din Wali, the patron saint of Kashmir.",
            "The original shrine was burned down in 1995 during a militant siege and later rebuilt.",
            "A major pilgrimage site visited by Muslims and Hindus alike.",
            "The walnut wood architecture is distinctive to Kashmiri shrine design.",
        ],
    },
    "Chhota_Imambara": {
        "display_name": "Chhota Imambara",
        "wikipedia_query": "Chhota Imambara",
        "location": "Lucknow, Uttar Pradesh",
        "state": "Uttar Pradesh",
        "lat": 26.8640,
        "lon": 80.9121,
        "opening_hours": "6:00 AM – 5:00 PM",
        "ticket_price": "₹25 (Indians) | ₹300 (Foreigners)",
        "category": "Imambara",
        "fun_facts": [
            "Built by Nawab Muhammad Ali Shah in 1838 as a congregation hall for Shia Muslims.",
            "Also called the 'Palace of Lights' due to its spectacular chandeliers imported from Belgium and Iran.",
            "Houses the golden throne of Muhammad Ali Shah and his own tomb.",
            "The central dome is flanked by two tall minarets and sits atop a raised platform.",
        ],
    },
    "Ellora Caves": {
        "display_name": "Ellora Caves",
        "wikipedia_query": "Ellora Caves",
        "location": "Aurangabad, Maharashtra",
        "state": "Maharashtra",
        "lat": 20.0258,
        "lon": 75.1780,
        "opening_hours": "6:00 AM – 6:00 PM (Closed on Tuesdays)",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Cave Monuments",
        "fun_facts": [
            "Contains 100 caves (34 open to public) representing Hindu, Buddhist, and Jain traditions.",
            "The Kailasa temple (Cave 16) is the world's largest rock-cut structure, carved from a single basalt cliff.",
            "Took over 150 years to build, starting around the 6th century CE.",
            "UNESCO World Heritage Site since 1983.",
        ],
    },
    "Fatehpur Sikri": {
        "display_name": "Fatehpur Sikri",
        "wikipedia_query": "Fatehpur Sikri",
        "location": "Agra District, Uttar Pradesh",
        "state": "Uttar Pradesh",
        "lat": 27.0945,
        "lon": 77.6706,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹55 (Indians) | ₹610 (Foreigners)",
        "category": "Mughal Palace Complex",
        "fun_facts": [
            "Built by Emperor Akbar in 1571 and served as the Mughal capital for only 14 years.",
            "Abandoned likely due to water scarcity; the entire complex was preserved as a ghost city.",
            "Buland Darwaza, the main gateway, is the tallest gateway in India at 54 metres.",
            "UNESCO World Heritage Site since 1986.",
        ],
    },
    "Gateway of India": {
        "display_name": "Gateway of India",
        "wikipedia_query": "Gateway of India",
        "location": "Colaba, Mumbai, Maharashtra",
        "state": "Maharashtra",
        "lat": 18.9220,
        "lon": 72.8347,
        "opening_hours": "Open 24 hours",
        "ticket_price": "Free",
        "category": "Arch Monument",
        "fun_facts": [
            "Built to commemorate the visit of King George V and Queen Mary to India in 1911.",
            "Constructed in Indo-Saracenic style using yellow basalt and reinforced concrete.",
            "The last British troops to leave India marched through this gate in February 1948.",
            "It overlooks the Arabian Sea directly opposite the Taj Mahal Palace Hotel.",
        ],
    },
    "Hawa mahal": {
        "display_name": "Hawa Mahal",
        "wikipedia_query": "Hawa Mahal",
        "location": "Jaipur, Rajasthan",
        "state": "Rajasthan",
        "lat": 26.9239,
        "lon": 75.8267,
        "opening_hours": "9:00 AM – 4:30 PM",
        "ticket_price": "₹50 (Indians) | ₹200 (Foreigners)",
        "category": "Palace",
        "fun_facts": [
            "Built in 1799 by Maharaja Sawai Pratap Singh, shaped like the crown of Lord Krishna.",
            "Has 953 small latticed windows (jharokhas) that create a natural air-conditioning effect.",
            "Designed so royal ladies could observe street festivals while remaining unseen (purdah).",
            "The facade is five storeys tall but only about 1.5 feet deep at its narrowest point.",
        ],
    },
    "Humayun_s Tomb": {
        "display_name": "Humayun's Tomb",
        "wikipedia_query": "Humayun's Tomb",
        "location": "Nizamuddin East, New Delhi",
        "state": "Delhi",
        "lat": 28.5933,
        "lon": 77.2507,
        "opening_hours": "6:00 AM – 6:00 PM",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Mughal Tomb",
        "fun_facts": [
            "The first garden-tomb in the Indian subcontinent, completed in 1565.",
            "Commissioned by Humayun's widow Hamida Banu Begum and designed by a Persian architect.",
            "Considered a direct architectural predecessor to the Taj Mahal.",
            "Over 150 Mughal family members are buried in smaller tombs within the complex.",
        ],
    },
    "India_gate": {
        "display_name": "India Gate",
        "wikipedia_query": "India Gate",
        "location": "Kartavya Path, New Delhi",
        "state": "Delhi",
        "lat": 28.6129,
        "lon": 77.2295,
        "opening_hours": "Open 24 hours",
        "ticket_price": "Free",
        "category": "War Memorial",
        "fun_facts": [
            "A memorial to 70,000 soldiers of the British Indian Army who died in World War I.",
            "Designed by architect Edwin Lutyens, standing 42 metres tall.",
            "Names of 13,300 soldiers are inscribed on the walls of the arch.",
            "The Amar Jawan Jyoti (eternal flame) burned at its base from 1972 until 2022.",
        ],
    },
    "Khajuraho": {
        "display_name": "Khajuraho Temples",
        "wikipedia_query": "Khajuraho Group of Monuments",
        "location": "Chhatarpur, Madhya Pradesh",
        "state": "Madhya Pradesh",
        "lat": 24.8518,
        "lon": 79.9199,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Hindu & Jain Temples",
        "fun_facts": [
            "The famous erotic carvings represent only ~10% of the sculptures; the rest depict daily life.",
            "Built between 950 and 1050 CE by the Chandela dynasty.",
            "Originally there were 85 temples; only 25 survive today.",
            "UNESCO World Heritage Site since 1986.",
        ],
    },
    "Sun Temple Konark": {
        "display_name": "Konark Sun Temple",
        "wikipedia_query": "Konark Sun Temple",
        "location": "Konark, Puri District, Odisha",
        "state": "Odisha",
        "lat": 19.8876,
        "lon": 86.0946,
        "opening_hours": "6:00 AM – 8:00 PM",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Hindu Temple",
        "fun_facts": [
            "Designed as a giant chariot of the sun god Surya, with 24 carved wheels and 7 horses.",
            "Built in the 13th century CE by King Narasimhadeva I of the Eastern Ganga dynasty.",
            "Known as the 'Black Pagoda' by European sailors who used it as a navigational landmark.",
            "UNESCO World Heritage Site since 1984.",
        ],
    },
    "alai_darwaza": {
        "display_name": "Alai Darwaza",
        "wikipedia_query": "Alai Darwaza",
        "location": "Qutb Complex, Mehrauli, New Delhi",
        "state": "Delhi",
        "lat": 28.5245,
        "lon": 77.1855,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners) — part of Qutb Complex",
        "category": "Gateway",
        "fun_facts": [
            "Built in 1311 by Sultan Alauddin Khalji as the southern gateway to the Quwwat-ul-Islam mosque.",
            "One of the earliest buildings in India to use true arch and true dome construction techniques.",
            "Decorated with ornate red sandstone inlaid with white marble geometric patterns.",
            "Considered a masterpiece of early Indo-Islamic architecture.",
        ],
    },
    "alai_minar": {
        "display_name": "Alai Minar",
        "wikipedia_query": "Alai Minar",
        "history": "The Alai Minar is an unfinished minaret in the Qutb complex, Mehrauli, Delhi, begun around 1311 CE by Sultan Alauddin Khalji of the Delhi Sultanate. Khalji planned it to be twice the height of the adjacent Qutb Minar, which would have made it the tallest structure in the medieval world at roughly 144 metres. Construction halted at the completion of the first storey after his death in 1316, leaving the massive rubble-masonry core, standing about 24.5 metres, without the sandstone cladding that was originally planned.",
        "location": "Qutb Complex, Mehrauli, New Delhi",
        "state": "Delhi",
        "lat": 28.5244,
        "lon": 77.1851,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners) — part of Qutb Complex",
        "category": "Unfinished Minaret",
        "fun_facts": [
            "Alauddin Khalji planned this minaret to be twice the height of the Qutb Minar (144 m total).",
            "Construction halted after his death in 1316, leaving only the 24.5 m first storey.",
            "Has the widest base of any minaret in India.",
            "The sandstone facing was never applied to the upper sections — still raw rubble masonry.",
        ],
    },
    "basilica_of_bom_jesus": {
        "display_name": "Basilica of Bom Jesus",
        "wikipedia_query": "Basilica of Bom Jesus",
        "location": "Old Goa, Goa",
        "state": "Goa",
        "lat": 15.5009,
        "lon": 73.9116,
        "opening_hours": "9:00 AM – 6:30 PM",
        "ticket_price": "Free",
        "category": "Baroque Church",
        "fun_facts": [
            "Holds the mortal remains of St. Francis Xavier, co-founder of the Jesuit order.",
            "Completed in 1605 — one of the oldest churches in India.",
            "The body of St. Francis Xavier is displayed in a silver casket and shown publicly every 10–12 years.",
            "UNESCO World Heritage Site since 1986 as part of 'Churches and Convents of Goa.'",
        ],
    },
    "charminar": {
        "display_name": "Charminar",
        "wikipedia_query": "Charminar",
        "location": "Hyderabad, Telangana",
        "state": "Telangana",
        "lat": 17.3616,
        "lon": 78.4747,
        "opening_hours": "9:30 AM – 5:30 PM",
        "ticket_price": "₹25 (Indians) | ₹300 (Foreigners)",
        "category": "Mosque & Monument",
        "fun_facts": [
            "Built in 1591 by Muhammad Quli Qutb Shah to commemorate the end of a plague epidemic.",
            "Its name literally means 'Four Minarets' — each minaret stands 56 metres tall.",
            "A mosque on the top floor has been in continuous use for over 400 years.",
            "The surrounding Laad Bazaar is one of India's oldest markets, famous for glass bangles.",
        ],
    },
    "golden temple": {
        "display_name": "Golden Temple (Harmandir Sahib)",
        "wikipedia_query": "Golden Temple",
        "location": "Amritsar, Punjab",
        "state": "Punjab",
        "lat": 31.6200,
        "lon": 74.8765,
        "opening_hours": "Open 24 hours, 7 days a week",
        "ticket_price": "Free",
        "category": "Sikh Gurdwara",
        "fun_facts": [
            "Officially called Sri Harmandir Sahib; the upper floors are covered with approximately 400 kg of gold.",
            "The Langar (free community kitchen) serves 50,000–100,000 free meals every single day.",
            "Surrounded by Amrit Sarovar (Pool of Nectar), which gives Amritsar its name.",
            "The temple has four doors on all four sides — symbolising openness to people of all religions.",
        ],
    },
    "iron_pillar": {
        "display_name": "Iron Pillar of Delhi",
        "wikipedia_query": "Iron Pillar of Delhi",
        "location": "Qutb Complex, Mehrauli, New Delhi",
        "state": "Delhi",
        "lat": 28.5245,
        "lon": 77.1854,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners) — part of Qutb Complex",
        "category": "Iron Monument",
        "fun_facts": [
            "Cast around 375–415 CE during the Gupta period — over 1,600 years old.",
            "Made of 98% pure wrought iron and shows no significant rusting, puzzling metallurgists for decades.",
            "Stands 7.21 metres tall and weighs approximately 6 tonnes.",
            "Legend: if you can wrap your arms around it with your hands clasped behind your back, your wish comes true.",
        ],
    },
    "jamali_kamali_tomb": {
        "display_name": "Jamali Kamali Mosque & Tomb",
        "wikipedia_query": "Jamali Kamali mosque and tomb",
        "history": "The Jamali Kamali mosque and tomb is a 16th-century monument in the Mehrauli Archaeological Park, New Delhi, dating to the Lodi and early Mughal era. It enshrines the tomb of Shaikh Fazlullah, the Sufi poet known as Jamali, who served at the courts of Babur and Humayun and is regarded as one of the finest poets of his age. The tomb's interior is decorated with brilliant painted plasterwork in red, blue and green, with verses from Jamali's poetry inscribed on the walls. Buried alongside him is 'Kamali', whose identity — possibly a brother, disciple, or close companion — remains one of Delhi's enduring historical mysteries.",
        "location": "Mehrauli Archaeological Park, New Delhi",
        "state": "Delhi",
        "lat": 28.5175,
        "lon": 77.1765,
        "opening_hours": "6:00 AM – 6:00 PM (Closed on Fridays)",
        "ticket_price": "Free",
        "category": "Lodi-era Tomb",
        "fun_facts": [
            "Jamali was a Sufi poet who served in the courts of Babur and Humayun; his verses are inscribed inside.",
            "The interior ceiling features vivid painted plasterwork in blue, red, and green — unusually well preserved.",
            "The identity of 'Kamali' buried alongside Jamali remains a mystery to historians.",
            "Considered one of Delhi's most atmospheric and lesser-known historical sites.",
        ],
    },
    "lotus_temple": {
        "display_name": "Lotus Temple",
        "wikipedia_query": "Lotus Temple",
        "location": "Bahapur, New Delhi",
        "state": "Delhi",
        "lat": 28.5535,
        "lon": 77.2588,
        "opening_hours": "9:00 AM – 5:30 PM (Closed on Mondays)",
        "ticket_price": "Free",
        "category": "Bahá'í Temple",
        "fun_facts": [
            "Designed by Iranian-Canadian architect Fariborz Sahba and completed in 1986.",
            "Shaped like a blooming lotus with 27 marble-clad petals arranged in groups of three.",
            "One of the most visited buildings in the world, receiving over 10,000 visitors daily.",
            "Open to people of all religions — no religious images, idols, or sermons inside.",
        ],
    },
    "mysore_palace": {
        "display_name": "Mysore Palace",
        "wikipedia_query": "Mysore Palace",
        "location": "Mysuru, Karnataka",
        "state": "Karnataka",
        "lat": 12.3052,
        "lon": 76.6552,
        "opening_hours": "10:00 AM – 5:30 PM",
        "ticket_price": "₹100 (Indians) | ₹200 (Foreigners)",
        "category": "Royal Palace",
        "fun_facts": [
            "Official residence of the Wadiyar dynasty; the current building was completed in 1912 after a fire.",
            "Built in Indo-Saracenic style, blending Hindu, Muslim, Rajput, and Gothic architectural elements.",
            "Illuminated by 97,000 light bulbs every Sunday evening and on festival days.",
            "The third most visited monument in India after the Taj Mahal and the Red Fort.",
        ],
    },
    "qutub_minar": {
        "display_name": "Qutub Minar",
        "wikipedia_query": "Qutb Minar",
        "location": "Mehrauli, New Delhi",
        "state": "Delhi",
        "lat": 28.5245,
        "lon": 77.1855,
        "opening_hours": "Sunrise to Sunset",
        "ticket_price": "₹40 (Indians) | ₹600 (Foreigners)",
        "category": "Minaret",
        "fun_facts": [
            "The world's tallest brick minaret at 72.5 metres, with 379 stairs to the top.",
            "Construction began around 1193 CE by Qutb ud-Din Aibak, founder of the Delhi Sultanate.",
            "Has five distinct storeys — the first three are red sandstone, the top two are marble and sandstone.",
            "UNESCO World Heritage Site since 1993.",
        ],
    },
    "tajmahal": {
        "display_name": "Taj Mahal",
        "wikipedia_query": "Taj Mahal",
        "location": "Agra, Uttar Pradesh",
        "state": "Uttar Pradesh",
        "lat": 27.1751,
        "lon": 78.0421,
        "opening_hours": "Sunrise to Sunset (Closed on Fridays)",
        "ticket_price": "₹250 (Indians) | ₹1,300 (Foreigners) + ₹200 extra for mausoleum",
        "category": "Mughal Mausoleum",
        "fun_facts": [
            "Built by Shah Jahan between 1631 and 1653 in memory of his wife Mumtaz Mahal.",
            "Took 22 years and over 20,000 artisans from across Asia to complete.",
            "The four minarets lean slightly outward so they fall away from the tomb in an earthquake.",
            "One of the Seven Wonders of the World and a UNESCO World Heritage Site since 1983.",
        ],
    },
    "tanjavur temple": {
        "display_name": "Brihadeeswara Temple (Thanjavur)",
        "wikipedia_query": "Brihadeeswarar Temple",
        "location": "Thanjavur, Tamil Nadu",
        "state": "Tamil Nadu",
        "lat": 10.7828,
        "lon": 79.1317,
        "opening_hours": "6:00 AM – 12:30 PM and 4:00 PM – 8:30 PM",
        "ticket_price": "Free (active place of worship)",
        "category": "Hindu Temple (Dravidian Architecture)",
        "fun_facts": [
            "Built around 1010 CE by Raja Raja Chola I — over 1,000 years old and still an active temple.",
            "The vimana (tower) is 66 metres tall; its dome casts no shadow on the ground at noon.",
            "Built entirely of interlocking granite blocks without any mortar.",
            "Part of the 'Great Living Chola Temples' UNESCO World Heritage Site since 1987.",
        ],
    },
    "victoria memorial": {
        "display_name": "Victoria Memorial",
        "wikipedia_query": "Victoria Memorial, Kolkata",
        "location": "Kolkata, West Bengal",
        "state": "West Bengal",
        "lat": 22.5448,
        "lon": 88.3426,
        "opening_hours": "Museum: 10:00 AM – 6:00 PM (Closed Mon) | Gardens: 5:30 AM – 6:15 PM",
        "ticket_price": "₹30 (Indians) | ₹500 (Foreigners)",
        "category": "Memorial & Museum",
        "fun_facts": [
            "Built between 1906 and 1921 using white Makrana marble — the same quarry as the Taj Mahal.",
            "Designed by Sir William Emerson in a blend of British and Mughal architectural styles.",
            "Now a museum housing over 28,000 artifacts related to British colonial history.",
            "The rotating bronze Angel of Victory atop the dome weighs approximately 3 tonnes.",
        ],
    },
}

# ==========================================
# 2. Wikipedia summary fetcher (uses requests + Wikipedia REST API)
#    No extra packages needed — requests is already installed.
# ==========================================

import requests
from urllib.parse import quote

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
HEADERS = {"User-Agent": "SMAI-A3-MonumentApp/1.0 (academic project)"}


def fetch_wiki_summary(query, sentences=4):
    """
    Fetches a plain-text summary from the Wikipedia REST API.
    URL-encodes the title (handles commas, apostrophes, etc.).
    Retries once on transient failure. Returns None if both attempts fail.
    """
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
# 3. Build the metadata dict and save JSON
# ==========================================

metadata = {}
failed_wiki = []

print(f"\nProcessing {len(MONUMENT_STATIC)} monuments...\n")

for key, static in MONUMENT_STATIC.items():
    display = static["display_name"]
    print(f"  [{list(MONUMENT_STATIC.keys()).index(key) + 1:02d}/24] {display}", end=" ... ", flush=True)

    # Fetch history from Wikipedia REST API
    history = fetch_wiki_summary(static["wikipedia_query"])
    time.sleep(0.3)  # polite rate limiting

    if history is None:
        # Use hardcoded history if provided, otherwise join fun facts
        history = static.get("history") or " ".join(static["fun_facts"])
        failed_wiki.append(display)
        print("fallback")
    else:
        print("ok")

    google_maps_url = f"https://maps.google.com/?q={static['lat']},{static['lon']}"
    wiki_title = quote(static["wikipedia_query"].replace(" ", "_"), safe="()'")
    wikipedia_url = f"https://en.wikipedia.org/wiki/{wiki_title}"

    metadata[key] = {
        "display_name": display,
        "location": static["location"],
        "state": static["state"],
        "lat": static["lat"],
        "lon": static["lon"],
        "opening_hours": static["opening_hours"],
        "ticket_price": static["ticket_price"],
        "category": static["category"],
        "history": history,
        "fun_facts": static["fun_facts"],
        "google_maps": google_maps_url,
        "wikipedia_url": wikipedia_url,
    }

# ==========================================
# 4. Save to metadata/monuments_metadata.json
# ==========================================

output_dir = os.path.join(ROOT_DIR, "metadata")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "monuments_metadata.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(metadata)}/24 monuments written to:")
print(f"  {output_path}")

if failed_wiki:
    print(f"\nWikipedia fetch failed for {len(failed_wiki)} monument(s) — fallback text used:")
    for name in failed_wiki:
        print(f"  - {name}")
