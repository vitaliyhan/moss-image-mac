# Scene composition with overlapping boxes

MOSS-Image takes a JSON layout as its prompt. Boxes don't have to be separate: they can **overlap and nest**, which
lets you build a detailed scene in depth. A sign mounted on a facade, a cat on a bench, a chef behind a curtain,
a person walking in front of a parked bicycle, each placed exactly where you want it.

![Overlapping boxes drawn over the result](images/overlap_boxes.jpg)

| `standard` (4 steps, ~70 s at 1024×576) | `high` (50 steps, CFG 3, ~21 min) |
|---|---|
| ![standard](images/overlap_standard.jpg) | ![high](images/overlap_high.jpg) |

*11 overlapping boxes, same seed (7), base M4. Spec: [`examples/overlapping_scene.json`](../examples/overlapping_scene.json).*

## The example

| # | Box `[x0, y0, x1, y1]` | Element | Overlaps |
|---|---|---|---|
| 1 | `[40, 40, 700, 520]` | wooden ramen shop (background) | contains 3-8 |
| 2 | `[20, 20, 1004, 110]` | row of red lanterns | across the top of 1 and the street |
| 3 | `[600, 70, 680, 380]` | text: neon sign `'RAMEN'` | on the right edge of 1 |
| 4 | `[220, 170, 470, 330]` | indigo noren curtain | inside 1 |
| 5 | `[260, 230, 440, 420]` | chef behind the counter | behind/below 4 |
| 6 | `[80, 400, 520, 500]` | wooden bench | lower part of 1 |
| 7 | `[140, 360, 280, 450]` | sleeping cat | on top of 6 |
| 8 | `[520, 360, 640, 520]` | text: chalkboard `'TONKOTSU 980'` | over 1 and the street |
| 9 | `[640, 330, 900, 540]` | blue bicycle | against 1, behind 10 |
| 10 | `[760, 140, 980, 576]` | woman with clear umbrella (foreground) | in front of 9 and the street |
| 11 | `[0, 470, 1024, 576]` | wet cobblestone street | under everything |

Run it:

```bash
.venv/bin/python examples/overlapping_scene.py                   # standard
.venv/bin/python examples/overlapping_scene.py --quality high    # 50 steps
.venv/bin/python examples/overlapping_scene.py --spec my_scene.json --seed 3
```

It saves the image and a `_boxes.jpg` overlay to `outputs/`. From an MCP app, pass the same `caption` and `items` to
`generate_layout_image` with `width: 1024, height: 576`.

## Tips

- **Background first, details later.** List big containers (building, street, sky) before the things on or in front of
  them. Nested boxes read as "on / inside", and a box that overlaps another's edge reads as "in front of".
- **Captions say how elements relate.** "a cat curled up asleep **on the bench**", "a woman **in the foreground**",
  "lanterns strung **across the top of the shop front**". The boxes give position; the words give depth and contact.
- **One box per piece of text.** Quote the exact words, and make the box roughly the shape of the text (tall and narrow
  for a vertical sign). Small text in small boxes is the hardest case. If a word comes out misspelled, try another seed
  or enlarge its box; in this example seed 2026 spelled the menu "TONKOTTU" and seed 7 got it right.
- **Keep the scene `caption` about style.** Light, mood, medium, lens. Words written there can show up as text.
- **Use `large` sizes for busy scenes.** 1024×576 gives each element enough pixels; 10+ boxes on a `fast` canvas get cramped.
- **Iterate in `standard`, finish in `high`.** The boxes fix the layout in both modes; `high` adds finer texture
  (wood grain, fur, steam, reflections). Small details like a figure's exact pose can differ between modes even with
  the same seed, because the samplers differ.
