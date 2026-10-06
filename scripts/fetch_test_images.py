"""Download the test image set into images/ (skips files that already exist).

Every image is public domain, CC0, MIT, Apache-2.0 or CC-BY; the licence and
author are listed next to each URL. The images themselves are not committed.

    python scripts/fetch_test_images.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "images"
USER_AGENT = "img2gcode-test-fetch/1.0 (https://github.com/RowanRama/img2gcode)"

# (filename, url, licence, author)
IMAGES = [
    ('logo_flag_south_korea.svg',
     'https://upload.wikimedia.org/wikipedia/commons/0/09/Flag_of_South_Korea.svg',
     'Public domain', 'Government of the Republic of Korea; vector by Great Brightstar and others'),
    ('logo_flag_nepal.svg',
     'https://upload.wikimedia.org/wikipedia/commons/9/9b/Flag_of_Nepal.svg',
     'Public domain', 'Pumbaa80, Achim1999'),
    ('logo_flag_kenya.svg',
     'https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Kenya.svg',
     'Public domain', 'User:Pumbaa80'),
    ('logo_flag_canada.svg',
     'https://upload.wikimedia.org/wikipedia/commons/c/cf/Flag_of_Canada.svg',
     'Public domain', 'George F. G. Stanley (design); SVG by E Pluribus Anthony / User:Mzajac'),
    ('logo_dell.svg',
     'https://cdn.jsdelivr.net/npm/simple-icons@16.34.0/icons/dell.svg',
     'CC0-1.0 (Simple Icons)', 'Simple Icons contributors'),
    ('logo_hp.svg',
     'https://cdn.jsdelivr.net/npm/simple-icons@16.34.0/icons/hp.svg',
     'CC0-1.0 (Simple Icons)', 'Simple Icons contributors'),
    ('logo_flag_brazil.png',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/1920px-Flag_of_Brazil.svg.png',
     'Public domain', 'Raimundo Teixeira Mendes (design); Wikimedia Commons rendering'),
    ('logo_flag_kiribati.png',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Flag_of_Kiribati.svg/1920px-Flag_of_Kiribati.svg.png',
     'Public domain', 'User:Hanzlan'),
    ('art_fox_face.svg',
     'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/svg/emoji_u1f98a.svg',
     'Apache-2.0', 'Google (Noto Emoji)'),
    ('art_rocket.svg',
     'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/svg/emoji_u1f680.svg',
     'Apache-2.0', 'Google (Noto Emoji)'),
    ('art_hibiscus.svg',
     'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/svg/emoji_u1f33a.svg',
     'Apache-2.0', 'Google (Noto Emoji)'),
    ('art_house.svg',
     'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/House/Flat/house_flat.svg',
     'MIT', 'Microsoft Corporation (Fluent UI Emoji)'),
    ('art_car.svg',
     'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Automobile/Flat/automobile_flat.svg',
     'MIT', 'Microsoft Corporation (Fluent UI Emoji)'),
    ('art_watermelon.svg',
     'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Watermelon/Flat/watermelon_flat.svg',
     'MIT', 'Microsoft Corporation (Fluent UI Emoji)'),
    ('art_poster_wpa_bighorn.jpg',
     'https://thumb.wikimedia.org/wikipedia/commons/thumb/7/7f/The_national_parks_preserve_wild_life_LCCN98518597.jpg/1280px-The_national_parks_preserve_wild_life_LCCN98518597.jpg',
     'Public domain', 'Work Projects Administration (WPA Poster Collection, Library of Congress), 1936-1939'),
    ('art_woodblock_great_wave.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1920px-Tsunami_by_hokusai_19th_century.jpg',
     'Public domain', 'Katsushika Hokusai'),
    ('lineart_durer_melencolia.jpg',
     'https://thumb.wikimedia.org/wikipedia/commons/thumb/0/05/Melencolia_I_MET_DP815743.jpg/1280px-Melencolia_I_MET_DP815743.jpg',
     'CC0', 'Albrecht Dürer (1514 engraving, Metropolitan Museum of Art scan)'),
    ('lineart_durer_rhinoceros.jpg',
     'https://thumb.wikimedia.org/wikipedia/commons/thumb/3/3f/D%C3%BCrer%27s_Rhinoceros%2C_1515.jpg/1280px-D%C3%BCrer%27s_Rhinoceros%2C_1515.jpg',
     'Public domain', 'Albrecht Dürer (1515 woodcut)'),
    ('lineart_gray_brain_dissection.png',
     'https://upload.wikimedia.org/wikipedia/commons/6/6a/Gray753.png',
     'Public domain', "Henry Vandyke Carter (Gray's Anatomy, fig. 753)"),
    ('lineart_patent_bicycle.jpg',
     'https://thumb.wikimedia.org/wikipedia/commons/thumb/9/9b/Patent_Drawing_for_J._J._Hentz%27_Bicycle_-_NARA_-_6277798.jpg/1280px-Patent_Drawing_for_J._J._Hentz%27_Bicycle_-_NARA_-_6277798.jpg',
     'Public domain', 'J. J. Hentz (US patent 624,651, 1899; NARA scan)'),
    ('lineart_coloring_pony.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/1/17/Pony_for_coloring.jpg',
     'Public domain', 'USDA (US government coloring-book page)'),
    ('lineart_botanical_smilax.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/1/1d/Smilax_rotundifolia_line_drawing.jpg',
     'Public domain', 'N. L. Britton & A. Brown (Illustrated Flora, 1913, via USDA PLANTS)'),
    ('lineart_botanical_pinus.png',
     'https://thumb.wikimedia.org/wikipedia/commons/thumb/9/98/Pinus_taeda_drawing.png/1280px-Pinus_taeda_drawing.png',
     'Public domain', 'N. L. Britton & A. Brown (Illustrated Flora, 1913, via USDA-NRCS PLANTS)'),
    ('portrait_nasa_peggy_whitson.jpg',
     'https://images-assets.nasa.gov/image/JSC2001-03044/JSC2001-03044~large.jpg',
     'Public Domain (NASA)', 'NASA / Bill Stafford (JSC)'),
    ('portrait_nasa_guion_bluford.jpg',
     'https://images-assets.nasa.gov/image/S92-48766/S92-48766~large.jpg',
     'Public Domain (NASA)', 'NASA (JSC)'),
    ('portrait_nasa_mae_jemison.jpg',
     'https://images-assets.nasa.gov/image/s87-45893/s87-45893~large.jpg',
     'Public Domain (NASA)', 'NASA (JSC)'),
    ('portrait_migrant_mother.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Lange-MigrantMother02.jpg/1280px-Lange-MigrantMother02.jpg',
     'Public Domain (FSA/OWI, Library of Congress)', 'Dorothea Lange (1936)'),
    ('portrait_lincoln_gardner_1863.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/1280px-Abraham_Lincoln_O-77_matte_collodion_print.jpg',
     'Public Domain', 'Alexander Gardner (1863)'),
    ('portrait_einstein_schmutzer_1921.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/3/33/Albert_Einstein_1921_by_F_Schmutzer_%283x4_close_cropped%29.jpg',
     'Public Domain', 'Ferdinand Schmutzer (1921)'),
    ('noisy_hongkong_night_high_iso.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Hong_Kong_in_night_%28noise_version%29.jpg/1920px-Hong_Kong_in_night_%28noise_version%29.jpg',
     'CC0', 'Wilfredor'),
    ('noisy_fawn_in_grass.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/White_tailed_deer_fawn_in_grass_Sand_Lake_NWR_%2814694197190%29.jpg/1920px-White_tailed_deer_fawn_in_grass_Sand_Lake_NWR_%2814694197190%29.jpg',
     'CC BY 2.0', 'USFWS Mountain-Prairie (U.S. Fish and Wildlife Service)'),
    ('noisy_film_grain_night_street_1939.jpg',
     'https://tile.loc.gov/storage-services/service/pnp/fsa/8a11000/8a11300/8a11342v.jpg',
     'Public Domain (FSA/OWI, U.S. government work, no known restrictions)', 'Arthur Rothstein (Farm Security Administration), 1939'),
    ('noisy_iss_night_city_lights.jpg',
     'https://images-assets.nasa.gov/image/iss035e017357/iss035e017357~large.jpg',
     'Public Domain (NASA)', 'NASA / Expedition 35 crew (Chris Hadfield)'),
    ('noisy_bobcat_trailcam_grass.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/e/ea/Bobcat_%2851016673236%29.jpg',
     'Public Domain (USFWS)', 'USFWS Mountain-Prairie (U.S. Fish and Wildlife Service)'),
    ('noisy_crowd_march_on_washington_1963.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Civil_Rights_March_on_Washington%2C_D.C._%28A_crowd_of_marchers.%29_-_NARA_-_542042.tif/lossy-page1-1920px-Civil_Rights_March_on_Washington%2C_D.C._%28A_crowd_of_marchers.%29_-_NARA_-_542042.tif.jpg',
     'Public Domain (USIA / NARA 542042, U.S. government work)', 'Rowland Scherman, for the U.S. Information Agency'),
    ('noisy_jpeg_q10_harbor.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/PSNR-example-comp-10.jpg/1920px-PSNR-example-comp-10.jpg',
     'CC0', 'Weedwhacker128'),
    ('noisy_clean_owl_feathers.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/0/0c/Great_horned_owl_%286293628118%29.jpg',
     'Public Domain (USFWS)', 'U.S. Fish and Wildlife Service Headquarters'),
    ('noisy_clean_red_fox_snow.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/e/ed/Red_fox_%2813436497065%29.jpg',
     'Public Domain (USFWS)', 'Alaska Region U.S. Fish & Wildlife Service'),
]


def main() -> int:
    DEST.mkdir(exist_ok=True)
    failed = 0
    for name, url, licence, author in IMAGES:
        path = DEST / name
        if path.exists():
            continue
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                path.write_bytes(resp.read())
            print(f"got  {name}  ({licence}, {author})")
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"FAIL {name}: {exc}", file=sys.stderr)
    print(f"{len(IMAGES) - failed}/{len(IMAGES)} images in {DEST}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
