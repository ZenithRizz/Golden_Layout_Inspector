# Test images for the Upload button

Upload from **Live → Upload image**. Each image was checked against the station before shipping.
Red circles in `answers/` show where the defects were planted.

| Image | Mode | Expected result |
|---|---|---|
| `target_repair.jpg` | Printed target (markers): compared with the OpenRAM layout | **REPAIR**: bridge at bitline br_6/vdd (column 6) + particle over columns 6–7, both in the bitcell array; spare columns 6 and 7 used |
| `target_scrap.jpg` | Printed target | **SCRAP**: particle in the PRECHARGE periphery (killer, no spare) + particle in the array (column 7) |
| `target_pass.jpg` | Printed target | **PASS**: no defects |
| `die_sram_array_no_markers.jpg` | Any die (no markers): cell-to-cell check | **RE-INSPECT**: 2 bridges + 1 open in our SRAM array, photographed without markers; no layout, so engineer review |
| `die_other_memory.jpg` | Any die | **RE-INSPECT**: particle + bridge + open in a *different* memory design next to non-repeating periphery |
| `die_other_memory_clean.jpg` | Any die | **PASS**: same design, no defects; the periphery must not raise false alarms |
| `not_a_memory_array.jpg` | Any die | **Error**: "No repeating memory array found" (expected rejection) |

`answer_key.json` lists each defect's class, region, nets, rows/columns, severity and action as the station reported them.
Regenerate (or make more) with `python scripts/make_test_images.py --out test_images`.
