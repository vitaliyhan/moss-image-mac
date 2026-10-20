import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import moss_client
import moss_mcp


class TestCutoutPipeline(unittest.TestCase):
    def test_resolve_bg_color_presets_and_hex(self):
        hex_code, rgb, desc = moss_client._resolve_bg_color("white")
        self.assertEqual(hex_code, "#FFFFFF")
        self.assertEqual(rgb, (255, 255, 255))
        self.assertIn("#FFFFFF", desc)

        hex_g, rgb_g, _ = moss_client._resolve_bg_color("GREEN")
        self.assertEqual(hex_g, "#00FF00")
        self.assertEqual(rgb_g, (0, 255, 0))

        hex_custom, rgb_custom, _ = moss_client._resolve_bg_color("#1a2b3c")
        self.assertEqual(hex_custom, "#1A2B3C")
        self.assertEqual(rgb_custom, (26, 43, 60))

        with self.assertRaises(ValueError):
            moss_client._resolve_bg_color("not-a-color")

    def test_cutout_prompt_layout_and_text_modes(self):
        raw_json = moss_client.cutout_prompt(
            "an antique brass astrolabe with white enamel dial",
            width=768,
            height=768,
            bg_color="white",
            use_layout_box=True,
            margin=0.10,
        )
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["bbox_coord_space"], "abs_centralized")
        self.assertEqual(parsed["bbox_format"], "yxyx")
        self.assertEqual(len(parsed["items"]), 1)
        item = parsed["items"][0]
        self.assertEqual(item["label"], "object")
        self.assertIn("antique brass astrolabe", item["caption"])
        # For 768x768 with 0.10 margin: x0=77, y0=77, x1=691, y1=691 -> centre-origin [-307, -307, 307, 307]
        self.assertEqual(item["bbox"], [-307, -307, 307, 307])

        flat_prompt = moss_client.cutout_prompt(
            "a red origami crane",
            width=512,
            height=512,
            bg_color="green",
            use_layout_box=False,
        )
        self.assertNotIn('"bbox_coord_space"', flat_prompt)
        self.assertIn("#00FF00", flat_prompt)
        self.assertIn("zero cast shadows", flat_prompt)

        with self.assertRaises(ValueError):
            moss_client.cutout_prompt("", width=512, height=512)
        with self.assertRaises(ValueError):
            moss_client.cutout_prompt("valid object", width=512, height=512, margin=0.5)

    def test_extract_cutout_rgba_preserves_interior_white_and_strips_border_bg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "synthetic_owl.png")
            # Create a 128x128 white-background image
            img = Image.new("RGB", (128, 128), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Outer closed dark-cobalt body silhouette from (28, 28) to (100, 100)
            draw.ellipse([28, 28, 100, 100], fill=(25, 70, 160))
            # Pure white (#FFFFFF) interior chest/eyes inside the closed silhouette (48..80)
            draw.ellipse([48, 48, 80, 80], fill=(255, 255, 255))
            img.save(src_path)

            rgba_path = moss_client.extract_cutout_rgba(src_path, tolerance=28, feather=1)
            self.assertTrue(os.path.exists(rgba_path))

            out_img = Image.open(rgba_path)
            self.assertEqual(out_img.mode, "RGBA")

            # 1. Exterior corner pixel (2, 2) must be 100% transparent
            self.assertEqual(out_img.getpixel((2, 2))[3], 0)
            self.assertEqual(out_img.getpixel((125, 125))[3], 0)

            # 2. Dark cobalt ring pixel (36, 64) must be 100% opaque
            r, g, b, a = out_img.getpixel((36, 64))
            self.assertEqual(a, 255)
            self.assertEqual((r, g, b), (25, 70, 160))

            # 3. Interior pure white core pixel (64, 64) must remain 100% opaque white!
            rw, gw, bw, aw = out_img.getpixel((64, 64))
            self.assertEqual(aw, 255)
            self.assertEqual((rw, gw, bw), (255, 255, 255))

    def test_chroma_key_magenta_and_green_hole_removal_and_despill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for bg_name, bg_rgb, spill_rgb in [
                ("magenta", (255, 0, 255), (210, 120, 195)),
                ("green", (0, 255, 0), (140, 220, 110)),
            ]:
                src_path = os.path.join(tmpdir, f"ring_{bg_name}.png")
                img = Image.new("RGB", (128, 128), bg_rgb)
                draw = ImageDraw.Draw(img)
                # Outer brass ring with an enclosed chroma-key hole in the center AND a white highlight
                draw.ellipse([20, 20, 108, 108], fill=spill_rgb)
                draw.ellipse([24, 24, 104, 104], fill=(190, 155, 85))
                # Enclosed hole of background color inside the ring
                draw.ellipse([48, 48, 80, 80], fill=bg_rgb)
                # Bright white specular highlight on the brass ring body
                draw.rectangle([30, 60, 40, 68], fill=(255, 255, 255))
                img.save(src_path)

                rgba_path = moss_client.extract_cutout_rgba(src_path, tolerance=28, feather=0)
                out_img = Image.open(rgba_path)
                # Exterior corner is transparent
                self.assertEqual(out_img.getpixel((2, 2))[3], 0)
                # Enclosed hole center (64, 64) is ALSO transparent!
                self.assertEqual(out_img.getpixel((64, 64))[3], 0)
                # White highlight on the brass ring (35, 64) is 100% opaque white!
                self.assertEqual(out_img.getpixel((35, 64)), (255, 255, 255, 255))
                # Rim pixel (22, 64) has chroma spill suppressed
                rr, rg, rb, ra = out_img.getpixel((22, 64))
                self.assertGreater(ra, 0)
                if bg_name == "magenta":
                    self.assertLessEqual(min(rr, rb), rg + 2)
                else:
                    self.assertLessEqual(rg, max(rr, rb) + 2)

    def test_generate_cutout_end_to_end_mocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "artifact.png")

            def fake_generate(prompt, width=768, height=768, seed=-1, out="out.png", steps=4, **kw):
                im = Image.new("RGB", (width, height), (255, 255, 255))
                d = ImageDraw.Draw(im)
                d.rectangle([width // 4, height // 4, 3 * width // 4, 3 * height // 4], fill=(180, 40, 40))
                im.save(out)
                return out, 1.25

            with patch("moss_client.generate", side_effect=fake_generate):
                plain_path, rgba_path, secs = moss_client.generate_cutout(
                    "a crimson gemstone",
                    width=256,
                    height=256,
                    out=out_file,
                    transparent=True,
                )
                self.assertEqual(plain_path, out_file)
                self.assertIsNotNone(rgba_path)
                self.assertTrue(os.path.exists(rgba_path))
                self.assertGreaterEqual(secs, 1.25)

                rgba_img = Image.open(rgba_path)
                self.assertEqual(rgba_img.getpixel((5, 5))[3], 0)
                self.assertEqual(rgba_img.getpixel((128, 128))[3], 255)

                # Also verify transparent=False mode
                plain_only, none_rgba, _ = moss_client.generate_cutout(
                    "a crimson gemstone",
                    width=256,
                    height=256,
                    out=out_file,
                    transparent=False,
                )
                self.assertEqual(plain_only, out_file)
                self.assertIsNone(none_rgba)

    def test_mcp_generate_cutout_object_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_generate_cutout(object_prompt, width, height, seed, out, steps,
                                     bg_color, transparent, use_layout_box, tolerance,
                                     on_progress=None, cancel_event=None):
                im = Image.new("RGB", (width, height), (255, 255, 255))
                d = ImageDraw.Draw(im)
                d.ellipse([40, 40, width - 40, height - 40], fill=(40, 140, 90))
                im.save(out)
                rgba_out = str(Path(out).with_name(Path(out).stem + "_rgba.png"))
                moss_client.extract_cutout_rgba(out, out_path=rgba_out, tolerance=tolerance)
                return out, rgba_out, 2.5

            with patch.object(moss_mcp, "OUT_DIR", Path(tmpdir)), \
                 patch.object(moss_mcp, "_ensure_server", return_value=None), \
                 patch("moss_client.generate_cutout", side_effect=fake_generate_cutout):
                res = asyncio.run(
                    moss_mcp.generate_cutout_object(
                        object_prompt="a jade mechanical scarab",
                        bg_color="white",
                        transparent=True,
                        aspect_ratio="1:1",
                        size="fast",
                    )
                )
                self.assertEqual(len(res), 2)
                self.assertIn("Transparent RGBA cutout:", res[1])


if __name__ == "__main__":
    unittest.main()
