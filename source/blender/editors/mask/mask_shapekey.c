/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2012 Blender Foundation.
 * All rights reserved.
 *
 *
 * Contributor(s): Blender Foundation,
 *                 Campbell Barton
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/editors/mask/mask_shapekey.c
 *  \ingroup edmask
 */

#include <stdlib.h>

#include "BLI_utildefines.h"
#include "BLI_listbase.h"
#include "BLI_math.h"

#include "BKE_context.h"
#include "BKE_depsgraph.h"
#include "BKE_mask.h"
#include "BKE_report.h"

#include "DNA_object_types.h"
#include "DNA_mask_types.h"
#include "DNA_scene_types.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_api.h"
#include "WM_types.h"

#include "ED_mask.h"  /* own include */

#include "mask_intern.h"  /* own include */

static int mask_shape_key_insert_exec(bContext *C, wmOperator *UNUSED(op))
{
	Scene *scene = CTX_data_scene(C);
	const int frame = CFRA;
	Mask *mask = CTX_data_edit_mask(C);
	MaskLayer *masklay;
	int change = FALSE;

	for (masklay = mask->masklayers.first; masklay; masklay = masklay->next) {
		MaskLayerShape *masklay_shape;

		if (!ED_mask_layer_select_check(masklay)) {
			continue;
		}

		masklay_shape = BKE_mask_layer_shape_varify_frame(masklay, frame);
		BKE_mask_layer_shape_from_mask(masklay, masklay_shape);
		change = TRUE;
	}

	if (change) {
		WM_event_add_notifier(C, NC_MASK | ND_DATA, mask);
		DAG_id_tag_update(&mask->id, 0);

		return OPERATOR_FINISHED;
	}
	else {
		return OPERATOR_CANCELLED;
	}
}

void MASK_OT_shape_key_insert(wmOperatorType *ot)
{
	/* identifiers */
	ot->name = "Insert Shape Key";
	ot->description = "";
	ot->idname = "MASK_OT_shape_key_insert";

	/* api callbacks */
	ot->exec = mask_shape_key_insert_exec;
	ot->poll = ED_maskedit_mask_poll;

	/* flags */
	ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

static int mask_shape_key_clear_exec(bContext *C, wmOperator *UNUSED(op))
{
	Scene *scene = CTX_data_scene(C);
	const int frame = CFRA;
	Mask *mask = CTX_data_edit_mask(C);
	MaskLayer *masklay;
	int change = FALSE;

	for (masklay = mask->masklayers.first; masklay; masklay = masklay->next) {
		MaskLayerShape *masklay_shape;

		if (!ED_mask_layer_select_check(masklay)) {
			continue;
		}

		masklay_shape = BKE_mask_layer_shape_find_frame(masklay, frame);

		if (masklay_shape) {
			BKE_mask_layer_shape_unlink(masklay, masklay_shape);
			change = TRUE;
		}
	}

	if (change) {
		Scene *scene = CTX_data_scene(C);
		BKE_mask_evaluate(mask, CFRA, TRUE);

		WM_event_add_notifier(C, NC_MASK | ND_DATA, mask);
		DAG_id_tag_update(&mask->id, OB_RECALC_DATA);

		return OPERATOR_FINISHED;
	}
	else {
		return OPERATOR_CANCELLED;
	}
}

void MASK_OT_shape_key_clear(wmOperatorType *ot)
{
	/* identifiers */
	ot->name = "Clear Shape Key";
	ot->description = "";
	ot->idname = "MASK_OT_shape_key_clear";

	/* api callbacks */
	ot->exec = mask_shape_key_clear_exec;
	ot->poll = ED_maskedit_mask_poll;

	/* flags */
	ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

static int mask_shape_key_feather_reset_exec(bContext *C, wmOperator *UNUSED(op))
{
	Scene *scene = CTX_data_scene(C);
	const int frame = CFRA;
	Mask *mask = CTX_data_edit_mask(C);
	MaskLayer *masklay;
	int change = FALSE;

	for (masklay = mask->masklayers.first; masklay; masklay = masklay->next) {

		if (masklay->restrictflag & (MASK_RESTRICT_VIEW | MASK_RESTRICT_SELECT)) {
			continue;
		}

		if (masklay->splines_shapes.first) {
			MaskLayerShape *masklay_shape_reset;
			MaskLayerShape *masklay_shape;

			/* get the shapekey of the current state */
			masklay_shape_reset = BKE_mask_layer_shape_alloc(masklay, frame);
			/* initialize from mask - as if inseting a keyframe */
			BKE_mask_layer_shape_from_mask(masklay, masklay_shape_reset);

			for (masklay_shape = masklay->splines_shapes.first;
			     masklay_shape;
			     masklay_shape = masklay_shape->next)
			{

				if (masklay_shape_reset->tot_vert == masklay_shape->tot_vert) {
					int i_abs = 0;
					int i;
					MaskSpline *spline;
					MaskLayerShapeElem *shape_ele_src;
					MaskLayerShapeElem *shape_ele_dst;

					shape_ele_src = (MaskLayerShapeElem *)masklay_shape_reset->data;
					shape_ele_dst = (MaskLayerShapeElem *)masklay_shape->data;

					for (spline = masklay->splines.first; spline; spline = spline->next) {
						for (i = 0; i < spline->tot_point; i++) {
							MaskSplinePoint *point = &spline->points[i];

							if (MASKPOINT_ISSEL_ANY(point)) {
								/* TODO - nicer access here */
								shape_ele_dst->value[6] = shape_ele_src->value[6];
							}

							shape_ele_src++;
							shape_ele_dst++;

							i_abs++;
						}
					}

				}
				else {
					// printf("%s: skipping\n", __func__);
				}

				change = TRUE;
			}

			BKE_mask_layer_shape_free(masklay_shape_reset);
		}
	}

	if (change) {
		WM_event_add_notifier(C, NC_MASK | ND_DATA, mask);
		DAG_id_tag_update(&mask->id, 0);

		return OPERATOR_FINISHED;
	}
	else {
		return OPERATOR_CANCELLED;
	}
}

void MASK_OT_shape_key_feather_reset(wmOperatorType *ot)
{
	/* identifiers */
	ot->name = "Feather Reset Animation";
	ot->description = "Reset feather weights on all selected points animation values";
	ot->idname = "MASK_OT_shape_key_feather_reset";

	/* api callbacks */
	ot->exec = mask_shape_key_feather_reset_exec;
	ot->poll = ED_maskedit_mask_poll;

	/* flags */
	ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

/*
 * - loop over selected shapekeys.
 * - find firstsel/lastsel pairs.
 * - move these into a temp list.
 * - re-key all the original shapes.
 * - copy unselected values back from the original.
 * - free the original.
 */
static int mask_shape_key_rekey_exec(bContext *C, wmOperator *op)
{
	Scene *scene = CTX_data_scene(C);
	const int frame = CFRA;
	Mask *mask = CTX_data_edit_mask(C);
	MaskLayer *masklay;
	int change = FALSE;

	const short do_feather  = RNA_boolean_get(op->ptr, "feather");
	const short do_location = RNA_boolean_get(op->ptr, "location");

	for (masklay = mask->masklayers.first; masklay; masklay = masklay->next) {

		if (masklay->restrictflag & (MASK_RESTRICT_VIEW | MASK_RESTRICT_SELECT)) {
			continue;
		}

		/* we need at least one point selected here to bother re-interpolating */
		if (!ED_mask_layer_select_check(masklay)) {
			continue;
		}

		if (masklay->splines_shapes.first) {
			MaskLayerShape *masklay_shape;
			MaskLayerShape *masklay_shape_lastsel = NULL;

			for (masklay_shape = masklay->splines_shapes.first;
			     masklay_shape;
			     masklay_shape = masklay_shape->next)
			{
				MaskLayerShape *masklay_shape_a = NULL;
				MaskLayerShape *masklay_shape_b = NULL;

				/* find contiguous selections */
				if (masklay_shape->flag & MASK_SHAPE_SELECT) {
					if (masklay_shape_lastsel == NULL) {
						masklay_shape_lastsel = masklay_shape;
					}
					if ((masklay_shape->next == NULL) ||
					    (((MaskLayerShape *)masklay_shape->next)->flag & MASK_SHAPE_SELECT) == 0)
					{
						masklay_shape_a = masklay_shape_lastsel;
						masklay_shape_b = masklay_shape;
						masklay_shape_lastsel = NULL;
					}
				}

				/* we have a from<>to? - re-interpolate! */
				if (masklay_shape_a && masklay_shape_b) {
					ListBase shapes_tmp = {NULL, NULL};
					MaskLayerShape *masklay_shape_tmp;
					MaskLayerShape *masklay_shape_tmp_next;
					MaskLayerShape *masklay_shape_tmp_last = masklay_shape_b->next;
					MaskLayerShape *masklay_shape_tmp_rekey;

					/* move keys */
					for (masklay_shape_tmp = masklay_shape_a;
					     masklay_shape_tmp && (masklay_shape_tmp != masklay_shape_tmp_last);
					     masklay_shape_tmp = masklay_shape_tmp_next)
					{
						masklay_shape_tmp_next = masklay_shape_tmp->next;
						BLI_remlink(&masklay->splines_shapes, masklay_shape_tmp);
						BLI_addtail(&shapes_tmp, masklay_shape_tmp);
					}

					/* re-key, note: cant modify the keys here since it messes uop */
					for (masklay_shape_tmp = shapes_tmp.first;
					     masklay_shape_tmp;
					     masklay_shape_tmp = masklay_shape_tmp->next)
					{
						BKE_mask_layer_evaluate(masklay, masklay_shape_tmp->frame, TRUE);
						masklay_shape_tmp_rekey = BKE_mask_layer_shape_varify_frame(masklay, masklay_shape_tmp->frame);
						BKE_mask_layer_shape_from_mask(masklay, masklay_shape_tmp_rekey);
						masklay_shape_tmp_rekey->flag = masklay_shape_tmp->flag & MASK_SHAPE_SELECT;
					}

					/* restore unselected points and free copies */
					for (masklay_shape_tmp = shapes_tmp.first;
					     masklay_shape_tmp;
					     masklay_shape_tmp = masklay_shape_tmp_next)
					{
						/* restore */
						int i_abs = 0;
						int i;
						MaskSpline *spline;
						MaskLayerShapeElem *shape_ele_src;
						MaskLayerShapeElem *shape_ele_dst;

						masklay_shape_tmp_next = masklay_shape_tmp->next;

						/* we know this exists, added above */
						masklay_shape_tmp_rekey = BKE_mask_layer_shape_find_frame(masklay, masklay_shape_tmp->frame);

						shape_ele_src = (MaskLayerShapeElem *)masklay_shape_tmp->data;
						shape_ele_dst = (MaskLayerShapeElem *)masklay_shape_tmp_rekey->data;

						for (spline = masklay->splines.first; spline; spline = spline->next) {
							for (i = 0; i < spline->tot_point; i++) {
								MaskSplinePoint *point = &spline->points[i];

								/* not especially efficient but makes this easier to follow */
								SWAP(MaskLayerShapeElem, *shape_ele_src, *shape_ele_dst);

								if (MASKPOINT_ISSEL_ANY(point)) {
									if (do_location) {
										memcpy(shape_ele_dst->value, shape_ele_src->value, sizeof(float) * 6);
									}
									if (do_feather) {
										shape_ele_dst->value[6] = shape_ele_src->value[6];
									}
								}

								shape_ele_src++;
								shape_ele_dst++;

								i_abs++;
							}
						}

						BKE_mask_layer_shape_free(masklay_shape_tmp);
					}

					change = TRUE;
				}
			}

			/* re-evaluate */
			BKE_mask_layer_evaluate(masklay, frame, TRUE);
		}
	}

	if (change) {
		WM_event_add_notifier(C, NC_MASK | ND_DATA, mask);
		DAG_id_tag_update(&mask->id, 0);

		return OPERATOR_FINISHED;
	}
	else {
		return OPERATOR_CANCELLED;
	}
}

void MASK_OT_shape_key_rekey(wmOperatorType *ot)
{
	/* identifiers */
	ot->name = "Re-Key Points of Selected Shapes";
	ot->description = "Recalculate animation data on selected points for frames selected in the dopesheet";
	ot->idname = "MASK_OT_shape_key_rekey";

	/* api callbacks */
	ot->exec = mask_shape_key_rekey_exec;
	ot->poll = ED_maskedit_mask_poll;

	/* flags */
	ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

	/* properties */
	RNA_def_boolean(ot->srna, "location", TRUE, "Location", "");
	RNA_def_boolean(ot->srna, "feather", TRUE, "Feather", "");
}


/* *** Shape Key Utils *** */

void ED_mask_layer_shape_auto_key(MaskLayer *masklay, const int frame)
{
	MaskLayerShape *masklay_shape;

	masklay_shape = BKE_mask_layer_shape_varify_frame(masklay, frame);
	BKE_mask_layer_shape_from_mask(masklay, masklay_shape);
}

int ED_mask_layer_shape_auto_key_all(Mask *mask, const int frame)
{
	MaskLayer *masklay;
	int change = FALSE;

	for (masklay = mask->masklayers.first; masklay; masklay = masklay->next) {
		ED_mask_layer_shape_auto_key(masklay, frame);
		change = TRUE;
	}

	return change;
}

int ED_mask_layer_shape_auto_key_select(Mask *mask, const int frame)
{
	MaskLayer *masklay;
	int change = FALSE;

	for (masklay = mask->masklayers.first; masklay; masklay = masklay->next) {

		if (!ED_mask_layer_select_check(masklay)) {
			continue;
		}

		ED_mask_layer_shape_auto_key(masklay, frame);
		change = TRUE;
	}

	return change;
}

/******************** shape key cleanup operator *********************/

static int mask_shape_key_cleanup_exec(bContext *C, wmOperator *op)
{
	Mask *mask = CTX_data_edit_mask(C);
	MaskLayer *mask_layer = BKE_mask_layer_active(mask);
	MaskLayerShape *current_shape, *next_shape;
	int removed_count = 0;
	const float tolerance = RNA_float_get(op->ptr, "tolerance");
	const float tolerance_squared = tolerance * tolerance;
	int width, height;
	float side;

	if (mask_layer == NULL) {
		return OPERATOR_CANCELLED;
	}

	ED_mask_get_size(CTX_wm_area(C), &width, &height);
	side = max_ff(width, height);

	for (current_shape = mask_layer->splines_shapes.first;
	     current_shape;
	     current_shape = next_shape)
	{
		MaskLayerShape *previous_shape = current_shape->prev;
		MaskLayerShapeElem *previous_elements, *current_elements, *next_elements;
		int i;
		float interpolation_factor, inv_interpolation_factor;
		float average_error;

		next_shape = current_shape->next;

		if (previous_shape == NULL || next_shape == NULL) {
			continue;
		}

		previous_elements = (MaskLayerShapeElem *) previous_shape->data;
		current_elements = (MaskLayerShapeElem *) current_shape->data;
		next_elements = (MaskLayerShapeElem *) next_shape->data;

		if (previous_shape->tot_vert != current_shape->tot_vert ||
		    previous_shape->tot_vert != next_shape->tot_vert ||
		    current_shape->tot_vert != next_shape->tot_vert)
		{
			printf("%s: unexpected mistmatch between shapes vertices number.\n", __func__);
			continue;
		}

		interpolation_factor = (float)(current_shape->frame - previous_shape->frame) /
		                       (float)(next_shape->frame - previous_shape->frame);
		inv_interpolation_factor = 1.0f - interpolation_factor;

		average_error = 0.0f;
		for (i = 0; i < current_shape->tot_vert; i++) {
			int j;
			for (j = 0; j < MASK_OBJECT_SHAPE_ELEM_SIZE; j++) {
				float interpolated_value;
				float current_error;
				interpolated_value = inv_interpolation_factor * previous_elements[i].value[j] +
				                     interpolation_factor * next_elements[i].value[j];
				current_error = (current_elements[i].value[j] - interpolated_value) * side;
				average_error += current_error * current_error;
			}
		}
		average_error /= (float) current_shape->tot_vert;

		if (ELEM(current_shape->frame, 3, 9)) {
			printf("%d %f\n", current_shape->frame, average_error);
		}

		if (average_error < tolerance_squared) {
			BLI_remlink(&mask_layer->splines_shapes, current_shape);
			BKE_mask_layer_shape_free(current_shape);
			removed_count++;
		}
	}

	if (removed_count > 0) {
		Scene *scene = CTX_data_scene(C);
		BKE_mask_evaluate(mask, CFRA, TRUE);

		WM_event_add_notifier(C, NC_MASK | ND_DATA, mask);
		DAG_id_tag_update(&mask->id, 0);
	}

	BKE_reportf(op->reports, RPT_INFO, "Removed %d keys", removed_count);

	return OPERATOR_FINISHED;
}

void MASK_OT_shape_key_cleanup(wmOperatorType *ot)
{
	/* identifiers */
	ot->name = "Cleanup Shape Keys";
	ot->description = "Remove keyframes which doesn't have much affect on mask shape";
	ot->idname = "MASK_OT_shape_key_cleanup";

	/* api callbacks */
	ot->exec = mask_shape_key_cleanup_exec;
	ot->poll = ED_maskedit_mask_poll;

	/* flags */
	ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

	/* properties */
	RNA_def_float(ot->srna, "tolerance", 0.8f, -FLT_MAX, FLT_MAX,
	              "Tolerance", "Average distance in pixels which mask is allowed to "
	                           "slide off after removing shapekey",
	              -100.0f, 100.0f);
}
