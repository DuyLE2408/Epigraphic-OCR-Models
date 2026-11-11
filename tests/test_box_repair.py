"""Local repair tests using synthetic glyph strokes, never private images."""

from copy import deepcopy
import unittest

from PIL import Image, ImageDraw

from src.box_repair import repair_boxes
from src.box_postprocess import filter_and_sort, rules_from_indices, sort_boxes


def box(index, x, top, bottom, width=20, region='body'):
    return {'image_id': 'bia_1.001', 'patch_id': f'bia_1.001.{index:03d}',
            'ocr_index': index, 'bbox': [[x, top], [x + width, top], [x + width, bottom], [x, bottom]],
            'sino_nom_ocr': str(index), 'confidence': 0.8, 'region': region, 'review_flags': []}


def page(invert=False):
    return Image.new('RGB', (160, 240), 'black' if invert else 'white')


def glyphs(image, x, top, bottom, invert=False):
    draw = ImageDraw.Draw(image)
    ink = 'white' if invert else 'black'
    for y in range(top, bottom, 20):
        draw.rectangle((x + 4, y, x + 5, y + 12), fill=ink)
        draw.rectangle((x + 13, y, x + 14, y + 12), fill=ink)
        draw.line((x + 4, y + 6, x + 14, y + 6), fill=ink)


class BoxRepairTests(unittest.TestCase):
    def test_tilted_fragments_merge_and_all_remaining_boxes_are_renumbered(self):
        image = page()
        glyphs(image, 30, 20, 80)
        glyphs(image, 32, 100, 180)
        glyphs(image, 90, 20, 180)
        boxes = [box(1, 30, 20, 80), box(2, 31, 90, 180), box(3, 90, 20, 180)]
        boxes[0]['bbox'] = [[30, 20], [50, 20], [49, 80], [29, 80]]
        boxes[1]['bbox'] = [[31, 90], [51, 90], [54, 180], [34, 180]]
        result = filter_and_sort(boxes, image.size, rules_from_indices([], []), image=image)
        self.assertEqual(len(result['boxes']), 2)
        self.assertEqual(result['boxes'][0]['source_ocr_indices'], [1, 2])
        self.assertEqual([b['reading_order'] for b in result['boxes']], [1, 2])
        self.assertEqual([b['patch_id'] for b in result['boxes']], ['bia_1.001.001', 'bia_1.001.002'])

    def test_parallel_skewed_columns_are_not_grouped_as_one_reading_column(self):
        left, right = box(2, 20, 20, 220), box(1, 40, 10, 210)
        left['bbox'] = [[20, 20], [44, 20], [56, 220], [32, 220]]
        right['bbox'] = [[40, 10], [64, 10], [76, 210], [52, 210]]
        self.assertEqual([b['ocr_index'] for b in sort_boxes([right, left])], [2, 1])

    def test_single_isolated_tail_is_only_a_proposal_without_independent_boundary_support(self):
        image = page()
        glyphs(image, 30, 20, 100)
        glyphs(image, 90, 20, 180)
        original = box(1, 30, 20, 80)
        result = repair_boxes(image, [original, box(2, 90, 20, 180)], refine_margins=False)
        target = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertEqual(target['bbox'], original['bbox'])
        self.assertTrue(any(not c['applied'] and c['reason'] == 'weak_character_continuation'
                            for c in result['changes']))

    def test_rightmost_tail_after_gap_can_use_two_neighbour_boundaries(self):
        image = page()
        glyphs(image, 110, 20, 120)
        glyphs(image, 110, 145, 146)
        glyphs(image, 80, 20, 160)
        glyphs(image, 50, 20, 160)
        result = repair_boxes(image, [box(1, 110, 20, 120), box(2, 80, 20, 160), box(3, 50, 20, 160)])
        target = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertGreater(target['bbox'][2][1], 150)
        self.assertTrue(target['ocr_needs_refresh'])

    def test_extension_preserves_slight_column_slant(self):
        image = page()
        glyphs(image, 31, 20, 120)
        glyphs(image, 90, 20, 180)
        tilted = box(1, 30, 20, 80)
        tilted['bbox'] = [[30, 20], [50, 20], [52, 80], [32, 80]]
        result = repair_boxes(image, [tilted, box(2, 90, 20, 180)])
        target = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertGreater(target['bbox'][2][1], 100)
        self.assertGreater(target['bbox'][3][0], target['bbox'][0][0])
        self.assertEqual(target['source_boxes'][0]['bbox'], tilted['bbox'])

    def test_merges_aligned_fragments_for_both_ink_polarities(self):
        for invert in (False, True):
            with self.subTest(invert=invert):
                image = page(invert)
                glyphs(image, 30, 20, 80, invert)
                glyphs(image, 30, 100, 180, invert)
                boxes = [box(1, 30, 20, 80), box(2, 30, 90, 180)]
                original = deepcopy(boxes)
                before = image.tobytes()
                result = repair_boxes(image, boxes, refine_margins=False)
                self.assertEqual(len(result['boxes']), 1)
                merged = result['boxes'][0]
                self.assertEqual(merged['source_ocr_indices'], [1, 2])
                self.assertEqual(merged['sino_nom_ocr'], '12')
                self.assertEqual(merged['ocr_text_status'], 'joined_fragments_unverified')
                self.assertTrue(merged['ocr_needs_refresh'])
                self.assertEqual([b['bbox'] for b in merged['source_boxes']], [b['bbox'] for b in original])
                self.assertEqual(result['applied_count'], 1)
                self.assertEqual(boxes, original)
                self.assertEqual(image.tobytes(), before)

    def test_merge_chain_keeps_top_down_text_and_all_source_indices(self):
        image = page()
        glyphs(image, 30, 20, 200)
        boxes = [box(3, 30, 140, 200), box(1, 30, 20, 80), box(2, 30, 90, 130)]
        result = repair_boxes(image, boxes)
        self.assertEqual(len(result['boxes']), 1)
        self.assertEqual(result['boxes'][0]['source_ocr_indices'], [1, 2, 3])
        self.assertEqual(result['boxes'][0]['sino_nom_ocr'], '123')

    def test_no_merge_between_neighbour_columns_or_large_paragraph_gap(self):
        image = page()
        glyphs(image, 30, 20, 80)
        glyphs(image, 60, 90, 180)
        glyphs(image, 30, 150, 210)
        result = repair_boxes(image, [box(1, 30, 20, 80), box(2, 60, 90, 180), box(3, 30, 150, 210)])
        self.assertFalse(any(c['action'] == 'merge' for c in result['changes']))

    def test_headers_and_slanted_polygons_are_not_automatically_reshaped(self):
        image = page()
        glyphs(image, 30, 20, 180)
        header = box(1, 30, 20, 80, region='header')
        slanted = box(2, 30, 90, 180)
        slanted['bbox'][0][0] += 6
        result = repair_boxes(image, [header, slanted])
        self.assertEqual(result['boxes'], [header, slanted])
        self.assertEqual(result['applied_count'], 0)

    def test_extends_short_missing_tail_without_fabricating_ocr_text(self):
        image = page()
        glyphs(image, 30, 20, 120)
        glyphs(image, 90, 20, 180)
        boxes = [box(1, 30, 20, 80), box(2, 90, 20, 180)]
        result = repair_boxes(image, boxes, refine_margins=False)
        changed = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertGreater(changed['bbox'][2][1], 110)
        self.assertEqual(changed['sino_nom_ocr'], '1')
        self.assertEqual(changed['ocr_text_status'], 'partial_after_box_extension')
        self.assertEqual(changed['source_boxes'][0]['bbox'], boxes[0]['bbox'])
        self.assertEqual(result['ocr_refresh_count'], 1)

    def test_extends_missing_top_but_stops_before_blank_margin(self):
        image = page()
        glyphs(image, 30, 20, 140)
        glyphs(image, 90, 20, 180)
        result = repair_boxes(image, [box(1, 30, 60, 140), box(2, 90, 20, 180)])
        changed = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertEqual(changed['bbox'][0][1], 19)
        self.assertEqual(changed['bbox'][2][1], 140)

    def test_long_tail_is_only_proposed_not_applied(self):
        image = page()
        glyphs(image, 30, 20, 220)
        glyphs(image, 90, 20, 220)
        original = box(1, 30, 20, 80)
        result = repair_boxes(image, [original, box(2, 90, 20, 220)], refine_margins=False)
        unchanged = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertEqual(unchanged['bbox'], original['bbox'])
        self.assertTrue(any(not c['applied'] and c['reason'] == 'long_extension_needs_review'
                            for c in result['changes']))

    def test_explicit_excluded_region_blocks_extension_and_merging(self):
        image = page()
        glyphs(image, 30, 20, 180)
        boxes = [box(1, 30, 20, 80), box(2, 30, 90, 180)]
        result = repair_boxes(image, boxes, blocked_regions=[[30, 80, 50, 90]], refine_margins=False)
        self.assertFalse(any(c['applied'] for c in result['changes']))
        self.assertEqual([b['bbox'] for b in result['boxes']], [b['bbox'] for b in boxes])

    def test_uniform_page_and_isolated_speck_do_not_extend_box(self):
        image = page()
        self.assertEqual(repair_boxes(image, [box(1, 30, 20, 80)])['applied_count'], 0)
        glyphs(image, 30, 20, 80)
        ImageDraw.Draw(image).point((35, 84), fill='black')
        result = repair_boxes(image, [box(1, 30, 20, 80)], refine_margins=False)
        self.assertEqual(result['applied_count'], 0)

    def test_solid_separator_and_low_contrast_noise_are_not_followed(self):
        image = page()
        glyphs(image, 30, 20, 120)
        ImageDraw.Draw(image).line((30, 83, 50, 83), fill='black', width=2)
        result = repair_boxes(image, [box(1, 30, 20, 80)], refine_margins=False)
        self.assertEqual(result['applied_count'], 0)
        faded = Image.new('RGB', image.size, (128, 128, 128))
        faded.paste((140, 140, 140), (34, 20, 36, 72))
        result = repair_boxes(faded, [box(1, 30, 20, 80)])
        self.assertEqual(result['applied_count'], 0)
        self.assertEqual(result['unassessed'][0]['reason'], 'insufficient_pixel_evidence')

    def test_extension_cannot_cross_another_header_box(self):
        image = page()
        glyphs(image, 30, 20, 120)
        glyphs(image, 90, 20, 180)
        result = repair_boxes(image, [box(1, 30, 20, 80), box(2, 30, 100, 120, region='header'),
                                      box(3, 90, 20, 180)])
        original = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertEqual(original['bbox'][2][1], 80)
        self.assertTrue(any(c['reason'] == 'overlaps_another_box' for c in result['changes']))

    def test_filter_sort_repairs_before_numbering_and_retains_manual_exclusion(self):
        image = page()
        glyphs(image, 30, 20, 180)
        boxes = [box(1, 30, 20, 80), box(2, 30, 90, 180)]
        rules = rules_from_indices([], [])
        result = filter_and_sort(boxes, image.size, rules, image=image)
        self.assertEqual(len(result['boxes']), 1)
        self.assertEqual(result['boxes'][0]['patch_id'], 'bia_1.001.001')
        self.assertGreater(result['repair']['applied_count'], 0)
        disabled = filter_and_sort(boxes, image.size, {**rules, 'auto_repair': False}, image=image)
        self.assertEqual(len(disabled['boxes']), 2)
        self.assertEqual(disabled['repair']['status'], 'disabled')
        excluded = filter_and_sort(boxes, image.size, rules_from_indices([], [2]), image=image)
        self.assertFalse(any(c['applied'] and c['action'] != 'tighten_sides'
                             for c in excluded['repair']['changes']))
        self.assertTrue(all(2 not in b.get('source_ocr_indices', [b['ocr_index']]) for b in excluded['boxes']))

    def test_side_tightening_keeps_glyph_strokes_and_original_ocr_for_both_polarities(self):
        for invert in (False, True):
            with self.subTest(invert=invert):
                image = page(invert)
                glyphs(image, 30, 20, 80, invert)
                original = box(1, 30, 20, 80)
                result = repair_boxes(image, [original])
                target = result['boxes'][0]
                self.assertGreater(target['bbox'][0][0], 30)
                self.assertLess(target['bbox'][1][0], 50)
                self.assertLessEqual(target['bbox'][0][0], 34)
                self.assertGreaterEqual(target['bbox'][1][0], 44)
                self.assertEqual(target['sino_nom_ocr'], original['sino_nom_ocr'])
                self.assertEqual(target['source_boxes'][0]['bbox'], original['bbox'])
                self.assertTrue(any(c['action'] == 'tighten_sides' and c['applied']
                                    and c['removed_foreground_fraction'] <= 0.01 for c in result['changes']))

    def test_visible_strokes_on_both_side_edges_are_not_trimmed(self):
        image = page()
        draw = ImageDraw.Draw(image)
        for x in (30, 49):
            draw.line((x, 20, x, 79), fill='black', width=2)
        original = box(1, 30, 20, 80)
        result = repair_boxes(image, [original])
        self.assertFalse(any(c['action'] == 'tighten_sides' for c in result['changes']))

    def test_weak_tail_marks_are_flagged_without_deleting_or_replacing_ocr_characters(self):
        image = page()
        glyphs(image, 30, 20, 60)
        ImageDraw.Draw(image).point((35, 73), fill='black')
        original = box(1, 30, 20, 80)
        original['sino_nom_ocr'] = 'ABC'
        result = repair_boxes(image, [original], refine_margins=False)
        self.assertEqual(result['boxes'][0]['sino_nom_ocr'], 'ABC')
        self.assertEqual(result['boxes'][0]['bbox'], original['bbox'])
        self.assertEqual(result['tail_review_count'], 1)
        self.assertTrue(result['tail_reviews'][0]['requires_text_alignment'])
        self.assertEqual(result['tail_reviews'][0]['action'], 'review_only')

    def test_tail_review_uses_final_reading_number_instead_of_original_ocr_index(self):
        image = page()
        glyphs(image, 30, 20, 60)
        glyphs(image, 90, 20, 180)
        ImageDraw.Draw(image).point((35, 73), fill='black')
        result = filter_and_sort([box(1, 90, 20, 180), box(2, 30, 20, 80)], image.size,
                                 rules_from_indices([], []), image=image)
        left = result['boxes'][0]
        self.assertEqual(left['ocr_index'], 2)
        self.assertEqual(left['tail_review']['reading_order'], 1)
        review = next(r for r in result['repair']['tail_reviews'] if r['source_ocr_indices'] == [2])
        self.assertEqual(review['reading_order'], 1)
        self.assertEqual(left['sino_nom_ocr'], '2')

    def test_longer_continuation_requires_independent_peer_boundary_evidence(self):
        image = page()
        for x in (110, 80, 50):
            glyphs(image, x, 20, 180)
        seed = box(1, 110, 20, 100)
        result = repair_boxes(image, [seed, box(2, 80, 20, 180), box(3, 50, 20, 180)], refine_margins=False)
        target = next(b for b in result['boxes'] if b['ocr_index'] == 1)
        self.assertGreater(target['bbox'][2][1], 160)
        self.assertTrue(target['ocr_needs_refresh'])
        without_peers = repair_boxes(image, [seed], refine_margins=False)
        self.assertEqual(without_peers['boxes'][0]['bbox'], seed['bbox'])

    def test_weak_extended_cells_do_not_pass_just_because_they_contain_bright_pixels(self):
        image = page()
        glyphs(image, 30, 20, 80)
        draw = ImageDraw.Draw(image)
        for y in (82, 102):
            draw.rectangle((34, y, 35, y + 4), fill='black')
            draw.rectangle((42, y, 43, y + 4), fill='black')
        original = box(1, 30, 20, 80)
        result = repair_boxes(image, [original], refine_margins=False)
        self.assertEqual(result['boxes'][0]['bbox'], original['bbox'])
        self.assertFalse(any(c['applied'] for c in result['changes']))


if __name__ == '__main__':
    unittest.main()
