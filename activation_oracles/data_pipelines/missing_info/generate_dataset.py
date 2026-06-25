"""
Generate the missing-information eval dataset.

Tests whether the AO can distinguish identical tokens with different model states.

For each problem pair (complete vs incomplete):
  A: complete prompt + truncated reasoning + short neutral segment (teacher-forced)
  B: incomplete prompt + truncated reasoning (same cut as C, no teacher forcing)
  C: incomplete prompt + truncated reasoning + SAME neutral segment as A (teacher-forced)

The AO probes the last N tokens. For A and C, these are the SAME short neutral
segment (~10 tokens), but the model's internal state differs because it processed
different prompts and reasoning traces before reaching the probe point.

If the AO uses activations, A and C should differ. If it just reads tokens, they match.

Pipeline:
1. Define the problem pairs (complete gives dimensions, incomplete gives derived quantity)
2. Generate thinking traces via vLLM (Qwen3-8B, thinking enabled)
3. Use the judge (local vLLM via pipeline_utils.async_api_call) to pick short neutral segments
4. Build 3 conditions per problem per segment → entries
5. Save dataset JSON

Usage:
    source .env && .venv/bin/python data_pipelines/missing_info/generate_dataset.py --model Qwen/Qwen3-8B
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import anthropic
import vllm
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import add_model_arg, add_n_per_task_arg, async_api_call, model_dir_name, parse_json_response, vllm_gpu_util

PROBLEMS = [
    {
        "id": "pool_border",
        "complete": (
            "A rectangular swimming pool is 25 meters long and 10 meters wide. "
            "A 2-meter-wide concrete border surrounds the pool on all sides. "
            "What is the total area of just the concrete border?"
        ),
        "incomplete": (
            "A rectangular swimming pool has an area of 250 square meters. "
            "A 2-meter-wide concrete border surrounds the pool on all sides. "
            "What is the total area of just the concrete border?"
        ),
        "withheld_keywords": ["25 meters", "25m", "10 meters", "10m", "25 m", "10 m"],
        "missing_info": "The individual length and width of the pool (only total area given)",
    },
    {
        "id": "room_cats",
        "complete": (
            "How many cats can fit in my room? The room is a 10m × 8.15m rectangle. "
            "My fox sits in one corner. Cats cannot be within 5m of the fox. "
            "Each cat occupies 0.5 m². How many cats can fit outside the fox's zone?"
        ),
        "incomplete": (
            "How many cats can fit in my room? The room is a rectangle with total area 81.5 m². "
            "My fox sits in one corner. Cats cannot be within 5m of the fox. "
            "Each cat occupies 0.5 m². How many cats can fit outside the fox's zone?"
        ),
        "withheld_keywords": ["10m", "8.15m", "10 m", "8.15 m", "10 meters", "8.15"],
        "missing_info": "The room dimensions (length and width)",
    },
    {
        "id": "box_diagonal",
        "complete": (
            "A rectangular box has dimensions 3m × 4m × 12m. "
            "What is the length of the longest straight rod that can fit inside the box?"
        ),
        "incomplete": (
            "A rectangular box has a volume of 144 cubic meters. "
            "What is the length of the longest straight rod that can fit inside the box?"
        ),
        "withheld_keywords": ["3m", "4m", "12m", "3 m", "4 m", "12 m", "3 meters", "4 meters", "12 meters"],
        "missing_info": "The individual dimensions of the box (only volume given)",
    },
    {
        "id": "garden_fence",
        "complete": (
            "An L-shaped garden consists of a 10m × 8m rectangle with a 4m × 3m "
            "rectangular section removed from one corner. How many meters of fencing "
            "are needed to enclose the garden completely?"
        ),
        "incomplete": (
            "An L-shaped garden has a total area of 68 square meters. "
            "How many meters of fencing are needed to enclose the garden completely?"
        ),
        "withheld_keywords": ["10m", "8m", "4m", "3m", "10 m", "8 m", "4 m", "3 m"],
        "missing_info": "The specific dimensions of the L-shape",
    },
    {
        "id": "wrap_present",
        "complete": (
            "I need to wrap a rectangular present that is 40cm long, 30cm wide, and 20cm tall. "
            "I need 3cm of overlap on each seam for taping. "
            "What is the minimum area of wrapping paper I need?"
        ),
        "incomplete": (
            "I need to wrap a rectangular present with a total surface area of 5200 cm². "
            "I need 3cm of overlap on each seam for taping. "
            "What is the minimum area of wrapping paper I need?"
        ),
        "withheld_keywords": ["40cm", "30cm", "20cm", "40 cm", "30 cm", "20 cm"],
        "missing_info": "The individual dimensions of the present (only surface area given)",
    },
    {
        "id": "cone_volume",
        "complete": (
            "A cone has a base radius of 6cm and a slant height of 10cm. "
            "What is the volume of the cone? Give your answer in terms of pi."
        ),
        "incomplete": (
            "A cone has a lateral (side) surface area of 60π cm². "
            "What is the volume of the cone? Give your answer in terms of pi."
        ),
        "withheld_keywords": ["6cm", "6 cm", "10cm", "10 cm", "radius of 6", "height of 10"],
        "missing_info": "The base radius and slant height of the cone",
    },
    {
        "id": "triangle_perimeter",
        "complete": (
            "A triangle has sides of length 7cm, 10cm, and 13cm. "
            "What is the perimeter of the triangle, and what is its area using Heron's formula?"
        ),
        "incomplete": (
            "A triangle has an area of approximately 34.98 cm². "
            "What is the perimeter of the triangle, and what is its area using Heron's formula?"
        ),
        "withheld_keywords": ["7cm", "10cm", "13cm", "7 cm", "10 cm", "13 cm"],
        "missing_info": "The individual side lengths of the triangle",
    },
    {
        "id": "field_diagonal",
        "complete": (
            "A rectangular field is 120 meters long and 50 meters wide. "
            "A person walks diagonally from one corner to the opposite corner. "
            "How much shorter is the diagonal path compared to walking along two sides?"
        ),
        "incomplete": (
            "A rectangular field has an area of 6000 square meters. "
            "A person walks diagonally from one corner to the opposite corner. "
            "How much shorter is the diagonal path compared to walking along two sides?"
        ),
        "withheld_keywords": ["120 meters", "50 meters", "120m", "50m", "120 m", "50 m"],
        "missing_info": "The individual length and width of the field",
    },
    {
        "id": "paint_walls",
        "complete": (
            "A room is 5 meters long, 4 meters wide, and 3 meters tall. "
            "Each wall needs two coats of paint. One liter of paint covers 10 square meters. "
            "How many liters of paint are needed for all four walls (not the ceiling or floor)?"
        ),
        "incomplete": (
            "A room has a total wall surface area of 54 square meters. "
            "Each wall needs two coats of paint. One liter of paint covers 10 square meters. "
            "How many liters of paint are needed for all four walls (not the ceiling or floor)?"
        ),
        "withheld_keywords": ["5 meters", "4 meters", "3 meters", "5m", "4m", "3m", "5 m", "4 m", "3 m"],
        "missing_info": "The individual length, width, and height of the room (only total wall area given)",
    },
    {
        "id": "trapezoid_area",
        "complete": (
            "A trapezoid has parallel sides of 8cm and 14cm, and a height of 6cm. "
            "What is the area of the trapezoid, and what is its perimeter if the "
            "non-parallel sides are each 7cm long?"
        ),
        "incomplete": (
            "A trapezoid has a perimeter of 36cm. "
            "What is the area of the trapezoid, and what is its perimeter if the "
            "non-parallel sides are each 7cm long?"
        ),
        "withheld_keywords": ["8cm", "14cm", "6cm", "8 cm", "14 cm", "6 cm", "parallel sides of 8", "parallel sides of 14", "height of 6"],
        "missing_info": "The parallel side lengths and height of the trapezoid",
    },
    {
        "id": "cylinder_side",
        "complete": (
            "A cylindrical tank has a radius of 7 meters and a height of 10 meters. "
            "What is the area of just the curved side surface of the tank?"
        ),
        "incomplete": (
            "A cylindrical tank has a volume of 490 pi cubic meters. "
            "What is the area of just the curved side surface of the tank?"
        ),
        "withheld_keywords": ["7 meters", "10 meters", "7m", "10m", "7 m", "10 m", "radius of 7", "height of 10"],
        "missing_info": "The radius and height of the cylinder (only the volume is given)",
    },
    {
        "id": "right_triangle_hyp",
        "complete": (
            "A right triangle has legs of length 9 cm and 12 cm. "
            "What is the length of its hypotenuse?"
        ),
        "incomplete": (
            "A right triangle has an area of 54 square cm. "
            "What is the length of its hypotenuse?"
        ),
        "withheld_keywords": ["9 cm", "12 cm", "9cm", "12cm", "legs of length 9", "length 9", "length 12"],
        "missing_info": "The two leg lengths of the right triangle (only the area is given)",
    },
    {
        "id": "photo_frame",
        "complete": (
            "A rectangular photo is 18 cm wide and 24 cm tall. A 3 cm uniform frame "
            "surrounds it on all sides. What is the area of just the frame?"
        ),
        "incomplete": (
            "A rectangular photo has an area of 432 square cm. A 3 cm uniform frame "
            "surrounds it on all sides. What is the area of just the frame?"
        ),
        "withheld_keywords": ["18 cm", "24 cm", "18cm", "24cm", "18 wide", "24 tall"],
        "missing_info": "The width and height of the photo (only the area is given)",
    },
    {
        "id": "parking_spaces",
        "complete": (
            "A rectangular parking lot is 50 meters long and 24 meters wide. "
            "Each car space measures 2.5 m by 5 m. How many car spaces fit in the lot?"
        ),
        "incomplete": (
            "A rectangular parking lot has an area of 1200 square meters. "
            "Each car space measures 2.5 m by 5 m. How many car spaces fit in the lot?"
        ),
        "withheld_keywords": ["50 meters", "24 meters", "50m", "24m", "50 m", "24 m"],
        "missing_info": "The length and width of the lot (only the area is given)",
    },
    {
        "id": "aquarium_glass",
        "complete": (
            "An open-top aquarium is 80 cm long, 30 cm wide, and 40 cm tall. "
            "How much glass is needed for its five faces?"
        ),
        "incomplete": (
            "An open-top aquarium has a volume of 96000 cubic cm. "
            "How much glass is needed for its five faces?"
        ),
        "withheld_keywords": ["80 cm", "30 cm", "40 cm", "80cm", "30cm", "40cm"],
        "missing_info": "The length, width and height of the aquarium (only the volume is given)",
    },
    {
        "id": "plot_diagonal",
        "complete": (
            "A rectangular plot is 9 meters by 40 meters. "
            "How long is the straight diagonal across it?"
        ),
        "incomplete": (
            "A rectangular plot has an area of 360 square meters. "
            "How long is the straight diagonal across it?"
        ),
        "withheld_keywords": ["9 meters", "40 meters", "9m", "40m", "9 m", "40 m", "9 by 40"],
        "missing_info": "The length and width of the plot (only the area is given)",
    },
    {
        "id": "box_surface",
        "complete": (
            "A closed rectangular box is 30 cm long, 20 cm wide, and 10 cm tall. "
            "What is its total surface area?"
        ),
        "incomplete": (
            "A closed rectangular box has a volume of 6000 cubic cm. "
            "What is its total surface area?"
        ),
        "withheld_keywords": ["30 cm", "20 cm", "10 cm", "30cm", "20cm", "10cm"],
        "missing_info": "The length, width and height of the box (only the volume is given)",
    },
    {
        "id": "garden_path",
        "complete": (
            "A rectangular garden is 12 m long and 8 m wide. A 1 m wide path runs "
            "along the inside edge all the way around. What is the area of the path?"
        ),
        "incomplete": (
            "A rectangular garden has an area of 96 square meters. A 1 m wide path runs "
            "along the inside edge all the way around. What is the area of the path?"
        ),
        "withheld_keywords": ["12 m", "8 m", "12m", "8m", "12 meters", "8 meters"],
        "missing_info": "The length and width of the garden (only the area is given)",
    },
    {
        "id": "cone_total_area",
        "complete": (
            "A cone has a base radius of 6 cm and a height of 8 cm. "
            "What is its total surface area, including the base and the slanted side?"
        ),
        "incomplete": (
            "A cone has a volume of 96 pi cubic cm. "
            "What is its total surface area, including the base and the slanted side?"
        ),
        "withheld_keywords": ["6 cm", "8 cm", "6cm", "8cm", "radius of 6", "height of 8"],
        "missing_info": "The base radius and height of the cone (only the volume is given)",
    },
    {
        "id": "pool_lap",
        "complete": (
            "A rectangular pool is 25 m long and 12 m wide. A swimmer swims one full "
            "lap around the edge. How far do they swim?"
        ),
        "incomplete": (
            "A rectangular pool has a surface area of 300 square meters. A swimmer swims "
            "one full lap around the edge. How far do they swim?"
        ),
        "withheld_keywords": ["25 m", "12 m", "25m", "12m", "25 meters", "12 meters"],
        "missing_info": "The length and width of the pool (only the area is given)",
    },
    {
        "id": "tablecloth",
        "complete": (
            "A rectangular table is 1.5 m wide and 2 m long. A tablecloth hangs 0.25 m "
            "over every edge. What is the total area of the tablecloth?"
        ),
        "incomplete": (
            "A rectangular table has a top area of 3 square meters. A tablecloth hangs "
            "0.25 m over every edge. What is the total area of the tablecloth?"
        ),
        "withheld_keywords": ["1.5 m", "2 m", "1.5m", "2m", "1.5 meters", "2 meters", "1.5 wide", "2 long"],
        "missing_info": "The width and length of the table (only the area is given)",
    },
    # --- Expansion. Same recipe: the "complete" prompt gives every independent
    # dimension; the "incomplete" prompt replaces them with ONE derived aggregate
    # (area / volume / perimeter) that genuinely under-determines the answer, so a
    # model that secretly used the missing numbers is detectable. Only multi-DOF
    # shapes qualify (a circle/sphere is fixed by any single aggregate).
    {
        "id": "ladder_wall",
        "complete": (
            "A ladder leans against a vertical wall, reaching 12 m up the wall with "
            "its base 5 m from the wall. How long is the ladder?"
        ),
        "incomplete": (
            "A ladder leans against a vertical wall, forming a right triangle with the "
            "wall and ground whose area is 30 square meters. How long is the ladder?"
        ),
        "withheld_keywords": ["12 m", "5 m", "12m", "5m", "12 meters", "5 meters", "up the wall", "from the wall"],
        "missing_info": "The height reached and base distance (only the enclosed area is given)",
    },
    {
        "id": "tv_diagonal",
        "complete": (
            "A television screen is 48 inches wide and 27 inches tall. "
            "What is the length of its diagonal?"
        ),
        "incomplete": (
            "A television screen has an area of 1296 square inches. "
            "What is the length of its diagonal?"
        ),
        "withheld_keywords": ["48 inches", "27 inches", "48in", "27in", "48 in", "27 in", "48 wide", "27 tall"],
        "missing_info": "The width and height of the screen (only the area is given)",
    },
    {
        "id": "ramp_slope",
        "complete": (
            "A wheelchair ramp rises 1 m over a horizontal run of 12 m. "
            "What is the length of the sloped surface of the ramp?"
        ),
        "incomplete": (
            "A wheelchair ramp has a triangular cross-section with an area of 6 square meters. "
            "What is the length of the sloped surface of the ramp?"
        ),
        "withheld_keywords": ["1 m", "12 m", "1m", "12m", "1 meter", "12 meters", "rise", "run of 12"],
        "missing_info": "The rise and horizontal run (only the cross-sectional area is given)",
    },
    {
        "id": "open_box_volume",
        "complete": (
            "An open-top box is 20 cm long, 15 cm wide, and 8 cm tall. "
            "What is its internal volume?"
        ),
        "incomplete": (
            "An open-top box has a total inner surface area of 860 square cm. "
            "What is its internal volume?"
        ),
        "withheld_keywords": ["20 cm", "15 cm", "8 cm", "20cm", "15cm", "8cm"],
        "missing_info": "The length, width and height of the box (only the surface area is given)",
    },
    {
        "id": "window_trim",
        "complete": (
            "A rectangular window is 60 cm wide and 90 cm tall. "
            "How much trim is needed to go all the way around its perimeter?"
        ),
        "incomplete": (
            "A rectangular window has an area of 5400 square cm. "
            "How much trim is needed to go all the way around its perimeter?"
        ),
        "withheld_keywords": ["60 cm", "90 cm", "60cm", "90cm", "60 wide", "90 tall"],
        "missing_info": "The width and height of the window (only the area is given)",
    },
    {
        "id": "rug_diagonal",
        "complete": (
            "A rectangular rug is 4 m long and 3 m wide. "
            "What is the length of its diagonal?"
        ),
        "incomplete": (
            "A rectangular rug has an area of 12 square meters. "
            "What is the length of its diagonal?"
        ),
        "withheld_keywords": ["4 m", "3 m", "4m", "3m", "4 meters", "3 meters", "4 long", "3 wide"],
        "missing_info": "The length and width of the rug (only the area is given)",
    },
    {
        "id": "triangle_heron2",
        "complete": (
            "A triangle has sides of length 6 cm, 8 cm, and 9 cm. "
            "What is its area using Heron's formula?"
        ),
        "incomplete": (
            "A triangle has a perimeter of 23 cm. "
            "What is its area using Heron's formula?"
        ),
        "withheld_keywords": ["6 cm", "8 cm", "9 cm", "6cm", "8cm", "9cm"],
        "missing_info": "The individual side lengths of the triangle (only the perimeter is given)",
    },
    {
        "id": "crate_diagonal",
        "complete": (
            "A rectangular crate is 6 m long, 3 m wide, and 2 m tall. "
            "What is the longest straight pole that can fit inside it?"
        ),
        "incomplete": (
            "A rectangular crate has a volume of 36 cubic meters. "
            "What is the longest straight pole that can fit inside it?"
        ),
        "withheld_keywords": ["6 m", "3 m", "2 m", "6m", "3m", "2m"],
        "missing_info": "The three dimensions of the crate (only the volume is given)",
    },
    {
        "id": "silo_surface",
        "complete": (
            "A cylindrical silo has a radius of 4 m and a height of 9 m. "
            "What is its total surface area including the top and bottom? Give your answer in terms of pi."
        ),
        "incomplete": (
            "A cylindrical silo has a volume of 144 pi cubic meters. "
            "What is its total surface area including the top and bottom? Give your answer in terms of pi."
        ),
        "withheld_keywords": ["4 m", "9 m", "4m", "9m", "radius of 4", "height of 9"],
        "missing_info": "The radius and height of the cylinder (only the volume is given)",
    },
    {
        "id": "funnel_cone",
        "complete": (
            "A conical funnel has a base radius of 3 cm and a height of 4 cm. "
            "What is its volume? Give your answer in terms of pi."
        ),
        "incomplete": (
            "A conical funnel has a lateral (side) surface area of 15 pi square cm. "
            "What is its volume? Give your answer in terms of pi."
        ),
        "withheld_keywords": ["3 cm", "4 cm", "3cm", "4cm", "radius of 3", "height of 4"],
        "missing_info": "The base radius and height of the cone (only the lateral surface area is given)",
    },
    {
        "id": "billboard_frame",
        "complete": (
            "A rectangular billboard is 8 m wide and 4 m tall. A 0.5 m frame surrounds "
            "it on all sides. What is the area of just the frame?"
        ),
        "incomplete": (
            "A rectangular billboard has an area of 32 square meters. A 0.5 m frame surrounds "
            "it on all sides. What is the area of just the frame?"
        ),
        "withheld_keywords": ["8 m", "4 m", "8m", "4m", "8 wide", "4 tall"],
        "missing_info": "The width and height of the billboard (only the area is given)",
    },
    {
        "id": "courtyard_path",
        "complete": (
            "A rectangular courtyard is 20 m long and 15 m wide. A 2 m wide path runs "
            "along the inside edge all the way around. What is the area of the path?"
        ),
        "incomplete": (
            "A rectangular courtyard has an area of 300 square meters. A 2 m wide path runs "
            "along the inside edge all the way around. What is the area of the path?"
        ),
        "withheld_keywords": ["20 m", "15 m", "20m", "15m", "20 long", "15 wide"],
        "missing_info": "The length and width of the courtyard (only the area is given)",
    },
    {
        "id": "parallelogram_area",
        "complete": (
            "A parallelogram has sides of 10 cm and 6 cm, with a height of 5 cm relative "
            "to the 10 cm side. What is its area?"
        ),
        "incomplete": (
            "A parallelogram has a perimeter of 32 cm. What is its area?"
        ),
        "withheld_keywords": ["10 cm", "6 cm", "5 cm", "10cm", "6cm", "5cm", "height of 5"],
        "missing_info": "The side lengths and height of the parallelogram (only the perimeter is given)",
    },
    {
        "id": "fish_tank_fill",
        "complete": (
            "A fish tank is 50 cm long, 25 cm wide, and 30 cm tall. "
            "How many liters of water fill it completely? (1 liter = 1000 cubic cm)"
        ),
        "incomplete": (
            "A fish tank has a rectangular base area of 1250 square cm. "
            "How many liters of water fill it completely? (1 liter = 1000 cubic cm)"
        ),
        "withheld_keywords": ["50 cm", "25 cm", "30 cm", "50cm", "25cm", "30cm", "height of 30", "30 tall"],
        "missing_info": "The height of the tank (only the base area is given, so depth is unknown)",
    },
    {
        "id": "picture_mat",
        "complete": (
            "A photo is 20 cm wide and 15 cm tall, mounted with a 4 cm mat border on all sides. "
            "What are the outer width and height of the matted picture?"
        ),
        "incomplete": (
            "A photo has an area of 300 square cm, mounted with a 4 cm mat border on all sides. "
            "What are the outer width and height of the matted picture?"
        ),
        "withheld_keywords": ["20 cm", "15 cm", "20cm", "15cm", "20 wide", "15 tall"],
        "missing_info": "The width and height of the photo (only the area is given)",
    },
    {
        "id": "roof_gable",
        "complete": (
            "A triangular roof gable has a base of 8 m and two equal sloped sides of 5 m each. "
            "What is its area?"
        ),
        "incomplete": (
            "A triangular roof gable has a perimeter of 18 m. What is its area?"
        ),
        "withheld_keywords": ["8 m", "5 m", "8m", "5m", "base of 8", "sides of 5"],
        "missing_info": "The base and sloped side lengths of the gable (only the perimeter is given)",
    },
    {
        "id": "box_surface2",
        "complete": (
            "A closed rectangular box is 24 cm long, 18 cm wide, and 12 cm tall. "
            "What is its total surface area?"
        ),
        "incomplete": (
            "A closed rectangular box has a volume of 5184 cubic cm. "
            "What is its total surface area?"
        ),
        "withheld_keywords": ["24 cm", "18 cm", "12 cm", "24cm", "18cm", "12cm"],
        "missing_info": "The length, width and height of the box (only the volume is given)",
    },
    {
        "id": "field_lap",
        "complete": (
            "A rectangular sports field is 100 m long and 64 m wide. "
            "How far is one full lap around its perimeter?"
        ),
        "incomplete": (
            "A rectangular sports field has an area of 6400 square meters. "
            "How far is one full lap around its perimeter?"
        ),
        "withheld_keywords": ["100 m", "64 m", "100m", "64m", "100 long", "64 wide"],
        "missing_info": "The length and width of the field (only the area is given)",
    },
]


def extract_thinking(response_text: str) -> str:
    """Extract content between <think> tags."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no closing tag, take everything after <think>
    match = re.search(r"<think>\s*(.*)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def _coerce_segment_list(parsed) -> list[str]:
    """Pull a flat list of neutral-segment strings out of whatever JSON shape the
    judge returned. The prompt asks for ["seg1", "seg2"], but the local Qwen
    judge is undisciplined, so we tolerate every common deviation:
      - list of strings                       -> as-is
      - list of {"neutral_segment": "..."}    -> pluck the field
      - dict wrapping the list under any key   -> first list value (recursed)
      - a single string                        -> [it]
    Anything else collapses to [], which the caller treats as "no usable points".
    """
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, dict):
        seg = parsed.get("neutral_segment")
        if isinstance(seg, str):
            return [seg]
        for v in parsed.values():
            if isinstance(v, list):
                return _coerce_segment_list(v)
        return []
    if isinstance(parsed, list):
        out = []
        for x in parsed:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) and isinstance(x.get("neutral_segment"), str):
                out.append(x["neutral_segment"])
        return out
    return []


def _derive_point(
    problem_id: str, segment: str, complete_reasoning: str,
    incomplete_reasoning: str, withheld_keywords: list[str],
) -> dict | None:
    """Turn ONE verbatim neutral segment into a full truncation point.

    The judge only supplies the short segment; we derive both prefixes locally
    (no fragile long-verbatim echo). Mechanism:
      1. reject segments containing a withheld keyword;
      2. locate the segment verbatim in the complete trace -> its position is the
         cut; complete_prefix = everything before it, snapped back to the last
         sentence boundary so it ends cleanly on .!?;
      3. incomplete_prefix = the incomplete trace cut at the sentence boundary
         nearest the SAME relative depth (the two traces share no text, so depth
         is the only sensible alignment).
    Returns None (with a one-line reason) if the segment is unusable.
    """
    segment = segment.strip()
    if len(segment.split()) < 2:
        print(f"  WARNING [{problem_id}]: segment too short, skipping: {segment!r}")
        return None
    if any(kw.lower() in segment.lower() for kw in withheld_keywords):
        print(f"  WARNING [{problem_id}]: segment has forbidden keyword, skipping: {segment!r}")
        return None

    seg_pos = complete_reasoning.find(segment)
    if seg_pos < 0:  # tolerate whitespace drift (newlines/spaces collapsed)
        collapsed = re.sub(r"\s+", " ", complete_reasoning)
        cseg = re.sub(r"\s+", " ", segment)
        ci = collapsed.find(cseg)
        seg_pos = -1 if ci < 0 else len(complete_reasoning[: _uncollapse_index(complete_reasoning, ci)])
    if seg_pos < 0:
        print(f"  WARNING [{problem_id}]: segment not found verbatim, skipping: {segment!r}")
        return None

    c_prefix = complete_reasoning[:seg_pos].rstrip()
    if not re.search(r"[.!?][\"')\]]?\s*$", c_prefix):  # snap to prior sentence end
        bounds = list(re.finditer(r"[.!?]\s", c_prefix))
        if not bounds:
            print(f"  WARNING [{problem_id}]: no sentence boundary before segment, skipping")
            return None
        c_prefix = c_prefix[: bounds[-1].end()].rstrip()

    target_ratio = len(c_prefix) / max(len(complete_reasoning), 1)
    target_pos = int(len(incomplete_reasoning) * target_ratio)
    bounds = [m.end() for m in re.finditer(r"[.!?]\s+", incomplete_reasoning)]
    i_prefix = (incomplete_reasoning[: min(bounds, key=lambda x: abs(x - target_pos))]
                if bounds else incomplete_reasoning[:target_pos]).rstrip()

    return {"complete_prefix": c_prefix, "incomplete_prefix": i_prefix, "neutral_segment": segment}


def _uncollapse_index(original: str, collapsed_idx: int) -> int:
    """Map an index in a whitespace-collapsed string back to the original."""
    seen = 0
    prev_space = False
    for i, ch in enumerate(original):
        if seen >= collapsed_idx:
            return i
        if ch.isspace():
            if not prev_space:
                seen += 1
            prev_space = True
        else:
            seen += 1
            prev_space = False
    return len(original)


async def find_truncation_points(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    problem_id: str,
    complete_reasoning: str,
    incomplete_reasoning: str,
    withheld_keywords: list[str],
) -> list[dict]:
    """Ask the judge for 2 short neutral segments, then derive truncation points.

    Design (robust to a weak local judge): we ask ONLY for two short verbatim
    sentence-starts from the complete trace — never the long prefixes. Echoing
    long, LaTeX-laden prefixes back as JSON strings is what broke the local Qwen
    judge (illegal backslashes, response truncation, refusals); short phrases
    sidestep all of it. The prefixes are reconstructed locally in _derive_point.

    Returns up to 2 dicts: {complete_prefix, incomplete_prefix, neutral_segment}.
    """
    async with semaphore:
        response = await async_api_call(
            client,
            model="claude-opus-4-6",
            max_tokens=1000,  # 2 short strings — no long verbatim echo
            messages=[{"role": "user", "content": f"""Below is a reasoning trace from a model solving a math problem. I need 2 short "neutral segments" from it for a teacher-forcing experiment.

<complete_reasoning>
{complete_reasoning}
</complete_reasoning>

A neutral segment is the VERBATIM beginning of a sentence (5 to 15 words), copied EXACTLY — character for character — from the reasoning above. It must be a generic reasoning phrase that carries NO problem-specific numbers, e.g. "So we need to figure out the total area." or "Now, let me reconsider the constraints here."

Pick exactly 2 segments:
- Segment 1 begins a sentence located roughly 30-50% through the trace.
- Segment 2 begins a sentence located roughly 50-80% through the trace.

Rules for each segment:
- It MUST be an exact substring of the reasoning above (copy it precisely, including punctuation).
- It MUST start at the beginning of a sentence and end at a word boundary.
- It MUST NOT contain any of these forbidden keywords (case-insensitive): {json.dumps(withheld_keywords)}

Return ONLY a JSON array of exactly 2 strings, nothing else, e.g.:
["So we need to figure out the total area.", "Now, let me reconsider the constraints here."]

Do NOT return objects, keys, prose, or an error — just the array of 2 strings."""}],
        )
        text = response.content[0].text
        try:
            segments = _coerce_segment_list(parse_json_response(text))
        except json.JSONDecodeError:
            print(f"  ERROR [{problem_id}]: failed to parse JSON response, first 500 chars:")
            print(f"    {text[:500]}")
            return []
        if not segments:
            print(f"  ERROR [{problem_id}]: no segments in response, first 500 chars:")
            print(f"    {text[:500]}")
            return []

        validated = [
            pt for seg in segments
            if (pt := _derive_point(problem_id, seg, complete_reasoning,
                                    incomplete_reasoning, withheld_keywords)) is not None
        ]
        if not validated:
            print(f"  ERROR [{problem_id}]: no valid truncation points found!")
        return validated[:2]


def _run_truncation_and_build(
    thinking_traces: dict, tokenizer: AutoTokenizer, model_name: str, output_path: Path,
) -> None:
    """Find truncation points via the judge, then build and save the dataset."""
    # The Anthropic client below is a no-op shim target: when AO_JUDGE_BASE_URL is
    # set (it is, for this task), async_api_call reroutes every call to the local
    # vLLM judge and ignores this client + the "claude-*" model arg. See
    # data_pipelines/pipeline_utils.py.
    judge = os.environ.get("AO_JUDGE_MODEL") or ("Anthropic API" if not os.environ.get("AO_JUDGE_BASE_URL") else "local-judge")
    print(f"\nFinding truncation points via the judge ({judge})...")
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(10)

    async def process_all():
        tasks = [
            find_truncation_points(
                client, semaphore, problem["id"],
                thinking_traces[problem["id"]]["complete"],
                thinking_traces[problem["id"]]["incomplete"],
                problem["withheld_keywords"],
            )
            for problem in PROBLEMS
        ]
        return await asyncio.gather(*tasks)

    all_truncation_points = asyncio.run(process_all())

    # Build dataset entries
    entries = []

    for problem, truncation_points in zip(PROBLEMS, all_truncation_points):
        for seg_idx, tp in enumerate(truncation_points):
            neutral_segment = tp["neutral_segment"]
            a_reasoning = tp["complete_prefix"]
            c_reasoning = tp["incomplete_prefix"]
            neutral_token_count = len(tokenizer.encode(neutral_segment, add_special_tokens=False))

            # Check for keyword leakage
            seg_lower = neutral_segment.lower()
            leaked_keywords = [kw for kw in problem["withheld_keywords"] if kw.lower() in seg_lower]

            suffix = f"_s{seg_idx}" if len(truncation_points) > 1 else ""

            print(f"\n  {problem['id']}{suffix}: neutral='{neutral_segment}' ({neutral_token_count} tokens)")
            if leaked_keywords:
                print(f"    WARNING: LEAKED KEYWORDS: {leaked_keywords}")

            # Condition A: complete prompt + truncated complete reasoning + neutral segment
            entries.append({
                "id": f"{problem['id']}{suffix}_A",
                "condition": "A_complete",
                "problem_id": problem["id"],
                "problem_text": problem["complete"],
                "full_reasoning": a_reasoning,
                "teacher_forced_segment": neutral_segment,
                "ground_truth_missing_info": False,
                "missing_info_description": problem["missing_info"],
                "neutral_segment": neutral_segment,
                "neutral_segment_token_count": neutral_token_count,
                "leaked_keywords": leaked_keywords,
            })

            # Condition B: incomplete prompt + same truncated reasoning as C (no teacher forcing)
            # B and C share the same truncated reasoning so the only difference
            # between B and C is the teacher-forced segment appended to C.
            entries.append({
                "id": f"{problem['id']}{suffix}_B",
                "condition": "B_incomplete",
                "problem_id": problem["id"],
                "problem_text": problem["incomplete"],
                "full_reasoning": c_reasoning,
                "teacher_forced_segment": "",
                "ground_truth_missing_info": True,
                "missing_info_description": problem["missing_info"],
                "neutral_segment": neutral_segment,
                "neutral_segment_token_count": neutral_token_count,
                "leaked_keywords": leaked_keywords,
            })

            # Condition C: incomplete prompt + truncated incomplete reasoning + SAME neutral segment
            entries.append({
                "id": f"{problem['id']}{suffix}_C",
                "condition": "C_forced",
                "problem_id": problem["id"],
                "problem_text": problem["incomplete"],
                "full_reasoning": c_reasoning,
                "teacher_forced_segment": neutral_segment,
                "ground_truth_missing_info": True,
                "missing_info_description": problem["missing_info"],
                "neutral_segment": neutral_segment,
                "neutral_segment_token_count": neutral_token_count,
                "leaked_keywords": leaked_keywords,
            })

    dataset = {
        "metadata": {
            "model": model_name,
            "total_entries": len(entries),
            "num_problems": len(PROBLEMS),
            "conditions": ["A_complete", "B_incomplete", "C_forced"],
            "description": (
                "Missing information experiment. Reasoning is truncated at a natural "
                "point, followed by a short teacher-forced neutral segment (~10 tokens). "
                "A and C have identical teacher-forced segments but different reasoning "
                "contexts (complete vs incomplete). If AO uses activations, A and C "
                "responses should differ."
            ),
        },
        "entries": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    leaked = sum(1 for e in entries if e["condition"] == "A_complete" and e["leaked_keywords"])
    n_a = sum(1 for e in entries if e["condition"] == "A_complete")
    print(f"\n{'='*60}")
    print(f"Saved {len(entries)} entries ({n_a} segments × 3 conditions)")
    print(f"Entries with keyword leakage: {leaked}/{n_a}")
    print(f"Output: {output_path}")

    # Print sample entries
    for e in entries[:3]:
        print(f"\n  {e['id']}:")
        print(f"    reasoning: {len(e['full_reasoning'])} chars")
        print(f"    teacher_forced: '{e['teacher_forced_segment']}'")
        print(f"    neutral_tokens: {e['neutral_segment_token_count']}")


def main(model_name: str):
    output_path = Path("data_pipelines/missing_info/missing_info_eval_dataset.json")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = vllm.LLM(
        model=model_name,
        max_model_len=8192,
        # GatedDeltaNet hybrid (Qwen3.5): eager decode is ~100x slower than with
        # CUDA graphs because the recurrent/conv path is launch-bound.
        enforce_eager=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm_gpu_util(0.7),
    )

    # Generate thinking traces for both complete and incomplete versions.
    # max_tokens must be large enough that the trace finishes its reasoning and
    # ends at a natural sentence boundary; the old 2000 cap cut Qwen3.5-4B's
    # traces mid-word, which then made the truncation-point judge (which must
    # find a sentence boundary + copy the next sentence verbatim) refuse or fail.
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=6000)

    all_prompts = []
    prompt_labels = []

    for problem in PROBLEMS:
        for version in ("complete", "incomplete"):
            messages = [{"role": "user", "content": problem[version]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            prompt_labels.append((problem["id"], version))

    print(f"Generating thinking traces for {len(all_prompts)} prompts...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Parse thinking traces
    thinking_traces: dict[str, dict[str, str]] = {}
    for (problem_id, version), output in zip(prompt_labels, outputs):
        raw = output.outputs[0].text
        thinking = extract_thinking(raw)
        if problem_id not in thinking_traces:
            thinking_traces[problem_id] = {}
        thinking_traces[problem_id][version] = thinking
        token_count = len(tokenizer.encode(thinking, add_special_tokens=False))
        print(f"  {problem_id}/{version}: {token_count} thinking tokens")

    # Free GPU memory
    del llm

    _run_truncation_and_build(thinking_traces, tokenizer, model_name, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate missing-information eval dataset")
    add_model_arg(parser)
    # Accepted for a uniform generator CLI but a NO-OP here: the dataset size is
    # fixed by the hand-written PROBLEMS list (N pairs × ≤2 segments × 3
    # conditions). Growing it means authoring more entries in PROBLEMS, not a flag.
    add_n_per_task_arg(parser)
    args = parser.parse_args()
    main(args.model)
