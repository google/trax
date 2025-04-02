# coding=utf-8
# Copyright 2022 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TensorFlow data sources and associated prepocessing functions."""

import itertools
import json
import math
import os
import random
import re

import gin
import numpy as np
import scipy
import scipy.special
import tensorflow as tf

# pylint: enable=invalid-name


def compute_single_result(op_name, num_args):
    """An implementation of the most popular ops from the MathQA dataset."""
    # See https://gitlab.cs.washington.edu/amini91/mathqa-categorization/
    # and specfically line 142 and following in new_DataStructure.py
    # for an implementation which covers more details.
    if op_name == "add":
        return num_args[0] + num_args[1]
    elif op_name == "circle_arc":
        return num_args[0] / 360 * math.pi * 2 * num_args[1]
    elif op_name == "circle_area":
        return math.pi * num_args[0] ** 2
    elif op_name == "circle_sector_area":
        return num_args[1] / 360 * math.pi * (num_args[0] ** 2)
    elif op_name == "circumface":
        return 2 * math.pi * num_args[0]
    elif op_name == "choose":
        return scipy.special.comb(num_args[0], num_args[1])
    elif op_name == "cosine":
        return math.cos(num_args[0])
    elif op_name == "cube_edge_by_volume":
        return num_args[0] ** (1 / 3)
    elif op_name == "combined_work":
        return 1 / (
            min(num_args[0], 1 / num_args[0]) + min(num_args[1], 1 / num_args[1])
        )
    elif op_name == "count_interval":
        return num_args[0] - num_args[1] + 1
    elif op_name == "diagonal":
        return math.sqrt(num_args[0] ** 2 + num_args[1] ** 2)
    elif op_name == "divide" or op_name == "speed":
        if num_args[1] != 0:
            return num_args[0] / num_args[1]
        else:
            return 0
    elif op_name == "factorial":
        return math.factorial(min(15, int(num_args[0])))
    elif op_name == "floor":
        return math.floor(num_args[0])
    elif op_name == "find_work":
        return 1 / (
            max(min(num_args[0], 1 / num_args[0]), min(num_args[1], 1 / num_args[1]))
            - min(min(num_args[0], 1 / num_args[0]), min(num_args[1], 1 / num_args[1]))
        )
    elif op_name == "from_percent":
        return num_args[0] / 100
    elif op_name == "gain_percent":
        return 100 + num_args[0]
    elif op_name == "gcd":
        return scipy.gcd(int(num_args[0]), int(num_args[1]))
    elif op_name == "inverse":
        if num_args[0] != 0:
            return 1 / num_args[0]
        else:
            return 0
    elif op_name == "lcm":
        return scipy.lcm(int(num_args[0]), int(num_args[1]))
    elif op_name == "log":
        return math.log(max(1e-5, num_args[0]), 2)
    elif op_name == "loss_percent":
        return 100 - num_args[0]
    elif op_name == "max":
        return max(num_args[0], num_args[1])
    elif op_name == "multiply":
        return num_args[0] * num_args[1]
    elif op_name == "negate_percent":
        return 100 - num_args[0]
    elif op_name == "negate":
        return -num_args[0]
    elif op_name == "original_price_before_loss":
        return num_args[1] * 100 / (100 + 1e-5 - num_args[0])
    elif op_name == "original_price_before_gain":
        return num_args[1] * 100 / (100 + num_args[0])
    elif op_name == "permutation":
        n, m = min(num_args[0], num_args[1]), max(num_args[0], num_args[1])
        return math.factorial(int(m)) / math.factorial(int(m - n))
    elif op_name == "power":
        return num_args[0] ** min(num_args[1], 5)
    elif op_name == "percent":
        return num_args[0] / 100 * num_args[1]
    elif op_name == "price_after_gain" or op_name == "p_after_gain":
        return (1 + num_args[0] / 100) * num_args[1]
    elif op_name == "price_after_loss" or op_name == "price_after_loss":
        return (1 - num_args[0] / 100) * num_args[1]
    elif op_name == "quadrilateral_area":
        return num_args[0] * (num_args[1] + num_args[2]) / 2
    elif op_name == "reminder":
        return num_args[0] % num_args[1]
    elif op_name == "rectangle_area":
        return num_args[0] * num_args[1]
    elif op_name == "rectangle_perimeter":
        return 2 * (num_args[0] + num_args[1])
    elif op_name == "rhombus_area":
        return num_args[0] * num_args[1] / 2
    elif op_name == "sine":
        return math.sin(num_args[0])
    elif op_name == "sqrt":
        return math.sqrt(max(0, num_args[0]))
    elif op_name == "subtract":
        return num_args[0] - num_args[1]
    elif op_name == "square_edge_by_perimeter":
        return num_args[0] / 4
    elif op_name == "square_edge_by_area":
        return math.sqrt(num_args[0])
    elif op_name == "square_area":
        return num_args[0] ** 2
    elif op_name == "surface_cube":
        return 6 * num_args[0] ** 2
    elif op_name == "surface_rectangular_prism":
        return 2 * (
            num_args[0] * num_args[1]
            + num_args[0] * num_args[2]
            + num_args[1] * num_args[2]
        )
    elif op_name == "semi_circle_perimiter":
        return math.pi * num_args[0] + 2 * num_args[0]
    elif op_name == "square_perimeter" or op_name == "rhombus_perimeter":
        return 4 * num_args[0]
    elif op_name == "surface_sphere":
        return 4 * math.pi * num_args[0] ** 2
    elif op_name == "speed_ratio_steel_to_stream":
        return (num_args[0] + num_args[1]) / (num_args[0] - num_args[1])
    elif op_name == "speed_in_still_water":
        return (num_args[0] + num_args[1]) / 2
    elif op_name == "stream_speed":
        return (num_args[0] - num_args[1]) / 2
    elif op_name == "trapezium_area":
        return num_args[0] * (num_args[1] + num_args[2]) / 2
    elif op_name == "triangle_area":
        return num_args[0] * num_args[1] / 2
    elif op_name == "triangle_perimeter":
        return num_args[0] + num_args[1] + num_args[2]
    elif op_name == "triangle_area_three_edges":
        # Heron's formula
        s = (num_args[0] + num_args[1] + num_args[2]) / 2
        return math.sqrt(
            max(0, s * (s - num_args[0]) * (s - num_args[1]) * (s - num_args[2]))
        )
    elif op_name == "union_prob":
        return num_args[0] + num_args[1] - num_args[2]
    elif op_name == "negate_prob":
        return 1 - num_args[0]
    elif op_name == "volume_cube":
        return num_args[0] ** 3
    elif op_name == "volume_cone":
        return math.pi * num_args[0] ** 2 * num_args[1] / 3
    elif op_name == "volume_cylinder":
        return math.pi * num_args[0] ** 2 * num_args[1]
    elif op_name == "volume_rectangular_prism":
        return num_args[0] * num_args[1] * num_args[2]
    elif op_name == "volume_sphere":
        return 4 / 3 * math.pi * num_args[0] ** 3


def compute_result(list_op, list_num):
    """Python execution of MathQA ops."""
    # The last of temporary results is the final answer.
    temporary_results = []
    for op in list_op:
        op_name = op.split("(")[0]
        start_bracket = op.find("(")
        end_bracket = op.find(")")
        op_args = op[start_bracket + 1 : end_bracket].split(",")
        num_args = []
        for arg in op_args:
            # The hash stands for a number stored in temporary_results.
            # For example #2 refers to the third temporary result.
            if arg[0] == "#":
                temp_index = int(
                    re.findall(
                        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", arg
                    )[0]
                )
                num_args.append(temporary_results[temp_index])
            # The n prefix stands for numbers which listed in list_num -
            # originally they were contained in the text.
            elif arg[0] == "n":
                n_index = int(
                    re.findall(
                        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", arg
                    )[0]
                )
                num_args.append(list_num[n_index])
            elif arg[0] == "c":
                if arg == "const_pi":
                    constant = math.pi
                elif arg == "const_deg_to_rad":
                    constant = math.pi / 180
                else:
                    consts = re.findall(
                        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", arg
                    )
                    if len(consts) == 1:
                        constant = float(consts[0])
                    else:
                        constant1 = float(consts[0])
                        constant2 = float("0." + consts[1])
                        constant = constant1 + constant2
                num_args.append(constant)
        temporary_results.append(compute_single_result(op_name, num_args))
    return temporary_results


def single_op_to_python_command(op_name, num_args):
    """An implementation of the most popular ops from the MathQA dataset."""
    # See https://gitlab.cs.washington.edu/amini91/mathqa-categorization/
    # and specfically line 142 and following in new_DataStructure.py
    # for an implementation which covers more details.
    if op_name == "add":
        return "{} + {}".format(num_args[0], num_args[1])
    elif op_name == "circle_arc":
        return "{} / 360 * math.pi * 2 * {}".format(num_args[0], num_args[1])
    elif op_name == "circle_area":
        return "math.pi * {}**2".format(num_args[0])
    elif op_name == "circle_sector_area":
        return "{} / 360 * math.pi * ({}**2)".format(num_args[1], num_args[0])
    elif op_name == "circumface":
        return "2 * math.pi * {}".format(num_args[0])
    elif op_name == "choose":
        return "scipy.special.comb({}, {})".format(num_args[0], num_args[1])
    elif op_name == "cosine":
        return "math.cos({})".format(num_args[0])
    elif op_name == "cube_edge_by_volume":
        return "{}**(1 / 3)".format(num_args[0])
    elif op_name == "combined_work":
        return "1 / (min({}, 1 / {}) + min({}, 1 / {}))".format(
            num_args[0], num_args[0], num_args[1], num_args[1]
        )
    elif op_name == "count_interval":
        return "{} - {} + 1".format(num_args[0], num_args[1])
    elif op_name == "diagonal":
        return "math.sqrt({}**2 + {}**2)".format(num_args[0], num_args[1])
    elif op_name == "divide" or op_name == "speed":
        # safe divide
        if num_args[1] != 0:
            return "{} / {}".format(num_args[0], num_args[1])
        else:
            return "0"
    elif op_name == "factorial":
        return "math.factorial(min(15, int({})))".format(num_args[0])
    elif op_name == "floor":
        return "math.floor({})".format(num_args[0])
    elif op_name == "find_work":
        return (
            "1 / (max(min({}, 1 / {}), min({}, 1 / {})) - min(min({}, 1 / {}), "
            "min({}, 1 / {})))"
        ).format(
            num_args[0],
            num_args[0],
            num_args[1],
            num_args[1],
            num_args[0],
            num_args[0],
            num_args[1],
            num_args[1],
        )
    elif op_name == "from_percent":
        return "{} / 100".format(num_args[0])
    elif op_name == "gain_percent":
        return "100 + {}".format(num_args[0])
    elif op_name == "gcd":
        return "scipy.gcd(int({}), int({}))".format(num_args[0], num_args[1])
    elif op_name == "inverse":
        # safe inverse
        if num_args[0] != 0:
            return "1 / {}".format(num_args[0])
        else:
            return "0"
    elif op_name == "lcm":
        return "scipy.lcm(int({}), int({}))".format(num_args[0], num_args[1])
    elif op_name == "log":
        return "math.log(max(1e-5, {}), 2)".format(num_args[0])
    elif op_name == "loss_percent":
        return "100 - {}".format(num_args[0])
    elif op_name == "max":
        return "max({},{})".format(num_args[0], num_args[1])
    elif op_name == "multiply":
        return "{} * {}".format(num_args[0], num_args[1])
    elif op_name == "negate_percent":
        return "100 - {}".format(num_args[0])
    elif op_name == "negate":
        return "-{}".format(num_args[0])
    elif op_name == "original_price_before_loss":
        return "{} * 100 / (100 + 1e-5 - {})  # original price before loss".format(
            num_args[1], num_args[0]
        )
    elif op_name == "original_price_before_gain":
        return "{} * 100 / (100 + {})  # original_price_before gain".format(
            num_args[1], num_args[0]
        )
    elif op_name == "permutation":
        return (
            "math.factorial(int(max({}, {}))) / math.factorial(int(max({}, {}) "
            "- min({}, {})))  # find all permutations"
        ).format(
            num_args[0], num_args[1], num_args[0], num_args[1], num_args[0], num_args[1]
        )
    elif op_name == "power":
        return "{}**min({}, 5)".format(num_args[0], num_args[1])
    elif op_name == "percent":
        return "{} / 100 * {}".format(num_args[0], num_args[1])
    elif op_name == "price_after_gain" or op_name == "p_after_gain":
        return "(1 + {} / 100) * {}".format(num_args[0], num_args[1])
    elif op_name == "price_after_loss" or op_name == "price_after_loss":
        return "(1 - {} / 100) * {}".format(num_args[0], num_args[1])
    elif op_name == "quadrilateral_area":
        return "{} * ({} + {}) / 2  # quadrilateral area".format(
            num_args[0], num_args[1], num_args[2]
        )
    elif op_name == "reminder":
        return "{} % {}".format(num_args[0], num_args[1])
    elif op_name == "rectangle_area":
        return "{} * {}  # area of rectangle".format(num_args[0], num_args[1])
    elif op_name == "rectangle_perimeter":
        return "2 * ({} + {})  # perimetere of rectangle".format(
            num_args[0], num_args[1]
        )
    elif op_name == "rhombus_area":
        return "{} * {} / 2".format(num_args[0], num_args[1])
    elif op_name == "sine":
        return "math.sin({})".format(num_args[0])
    elif op_name == "sqrt":
        return "math.sqrt(max(0, {}))".format(num_args[0])
    elif op_name == "subtract":
        return "{} - {}".format(num_args[0], num_args[1])
    elif op_name == "square_edge_by_perimeter":
        return "{} / 4. # square edge given perimeter".format(num_args[0])
    elif op_name == "square_edge_by_area":
        return "math.sqrt({})  # square edge given area".format(num_args[0])
    elif op_name == "square_area":
        return "{}**2".format(num_args[0])
    elif op_name == "surface_cube":
        return "6 * {}**2  # surface of a cube".format(num_args[0])
    elif op_name == "surface_rectangular_prism":
        return "2 * ({} * {} + {} * {} + {} * {})  # surface of a rectangular prism".format(
            num_args[0], num_args[1], num_args[0], num_args[2], num_args[1], num_args[2]
        )
    elif op_name == "semi_circle_perimiter":
        return "math.pi * {} + 2 * {}  # perimeter of a semi-circle".format(
            num_args[0], num_args[0]
        )
    elif op_name == "square_perimeter" or op_name == "rhombus_perimeter":
        return "4 * {}".format(num_args[0])
    elif op_name == "surface_sphere":
        return "4 * math.pi * {}**2".format(num_args[0])
    elif op_name == "speed_ratio_steel_to_stream":
        return "({} + {}) / ({} - {})".format(
            num_args[0], num_args[1], num_args[0], num_args[1]
        )
    elif op_name == "speed_in_still_water":
        return "{} + {} / 2".format(num_args[0], num_args[1])
    elif op_name == "stream_speed":
        return "{} - {} / 2".format(num_args[0], num_args[1])
    elif op_name == "trapezium_area":
        return "{} * ({} + {}) / 2".format(num_args[0], num_args[1], num_args[2])
    elif op_name == "triangle_area":
        return "{} * {} / 2".format(num_args[0], num_args[1])
    elif op_name == "triangle_perimeter":
        return "{} + {} + {}  # perimeter of a triangle".format(
            num_args[0], num_args[1], num_args[2]
        )
    elif op_name == "triangle_area_three_edges":
        return (
            "(lambda s, a, b, c: math.sqrt(max(0, s * (s - a) * (s - b) * (s - "
            "c))))(({} + {} + {}) / 2, {}, {}, {})  # Heron's formula"
        ).format(
            num_args[0], num_args[1], num_args[2], num_args[0], num_args[1], num_args[2]
        )
    elif op_name == "union_prob":
        return "{} + {} - {}".format(num_args[0], num_args[1], num_args[2])
    elif op_name == "negate_prob":
        return "1 - {}".format(num_args[0])
    elif op_name == "volume_cube":
        return "{}**3".format(num_args[0])
    elif op_name == "volume_cone":
        return "math.pi * {}**2 * {} / 3".format(num_args[0], num_args[1])
    elif op_name == "volume_cylinder":
        return "math.pi * {}**2 * {}".format(num_args[0], num_args[1])
    elif op_name == "volume_rectangular_prism":
        return "{} * {} * {}".format(num_args[0], num_args[1], num_args[2])
    elif op_name == "volume_sphere":
        return "4 / 3 * math.pi * {}**3".format(num_args[0])


def compute_program(list_op):
    """Python execution of MathQA ops."""
    # The last of temporary results is the final answer.
    temporary_results = []
    num_op = 0
    for op in list_op:
        op_name = op.split("(")[0]
        start_bracket = op.find("(")
        end_bracket = op.find(")")
        op_args = op[start_bracket + 1 : end_bracket].split(",")
        num_args = []
        for arg in op_args:
            # The hash stands for a number stored in temporary_results.
            # For example #2 refers to the third temporary result.
            if arg[0] == "#":
                temp_index = int(
                    re.findall(
                        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", arg
                    )[0]
                )
                num_args.append("t{}".format(temp_index))
            # The n prefix stands for numbers which listed in list_num -
            # originally they were contained in the text.
            elif arg[0] == "n":
                # n_index = int(
                #     re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                #                arg)[0])
                num_args.append(arg)
            elif arg[0] == "c":
                if arg == "const_pi":
                    constant = math.pi
                elif arg == "const_deg_to_rad":
                    constant = math.pi / 180
                else:
                    consts = re.findall(
                        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", arg
                    )
                    if len(consts) == 1:
                        constant = float(consts[0])
                    else:
                        constant1 = float(consts[0])
                        constant2 = float("0." + consts[1])
                        constant = constant1 + constant2
                num_args.append(str(constant))
        temporary_result = "t{} = {}".format(
            num_op, single_op_to_python_command(op_name, num_args)
        )
        temporary_results.append(temporary_result)
        num_op += 1
    return temporary_results


def compute_nums(question):
    """Finds numbers in a string and convert them to floats."""
    # The funny looking replace is needed to deal with numbers such as 4,000
    # TODO(henrykm) deal with numbers written as words "one", "two", ...
    return [
        float(num.replace(",", ""))
        for num in re.findall(
            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", question
        )
    ]


def compute_ops(linear_formula):
    list_op = linear_formula.split("|")
    # In some cases the list of operations contains a superflous last element,
    # namely an empty string.
    if not list_op[-1]:
        list_op = list_op[:-1]
    return list_op


def process_single_mathqa_example(example):
    """Execute a single example and verify coherence of a MathQA problem.

    Args:
      example: a dictionary with the following fields: Problem - a natural
        language formulation of the problem Rationale - a natural language
        solution of the problem options - five possible answers ( a) b) c) d) and
        e) ) correct - the letter representing the correct answer
        annotated_formula - formula representing the full solution linear_formula
        - a string of operations separated by the | character, e.g.
        multiply(n2,const_100)|multiply(n0,n1)|divide(#0,#1)|
        multiply(#2,const_100)|divide(#3,#1)| category - a natural language
        description of the category to which a given problem belongs.

    Returns:
      answer_num: numerical answer contained in the example
      python_result: numerical answers computed in Python, including intermediate
        results. The answer_num should be close python_result[-1]
      list_op: list of arithmetic operations
      list_num: list of identified numbers in the text
    """
    question = example["Problem"]
    list_num = compute_nums(question)
    list_op = compute_ops(example["linear_formula"])
    answers = example["options"]
    correct_answer = example["correct"]
    index = answers.find("{} )".format(correct_answer))
    answer_string = re.findall(
        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", answers[index:]
    )
    # The if statement deals with empty lists - they are needed to treat
    # a correct non-numerical answer e) None of the above. Here we do not want
    # non-numerical answers, hence we return None.
    if answer_string:
        answer_num = float(
            re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", answers[index:]
            )[0].replace(",", "")
        )
    else:
        return None
    # The if statements below deals with answers written as fractions e.g.
    # a ) 1 / 2 , b ) 1 / 3 , c ) 1 / 5 , d ) 10 / 30 , e ) 2 / 5 ?
    index_end_of_answer = index + len(str(answer_num)) + 3
    if index_end_of_answer < len(answers) and answers[index_end_of_answer] == "/":
        answer_denom = float(
            re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                answers[index_end_of_answer:],
            )[0].replace(",", "")
        )
        answer_num /= answer_denom
    python_result = compute_result(list_op, list_num)
    python_program = compute_program(list_op)
    return answer_num, python_result, python_program, list_op, list_num


def convert_float_to_mathqa(number):
    floor = int(float(number))
    if floor == number:
        return "const_" + str(floor)
    else:
        return "const_" + str(floor) + "_" + str(number)[len(str(floor)) + 1 :]


def convert_to_subtract(const_string):
    return "subtract({},const_0)".format(const_string)


def execute_mathqa_dsl_program(problem, dsl_code):
    """Executes the DSL code for a given problem.

    Args:
      problem: problem formulation (needed to get parameters).
      dsl_code: DSL code.

    Returns:
      the result of executing of the DSL code.
    """
    n0_loc = problem.find("n0")
    list_num = compute_nums(problem[n0_loc:])
    # The list contains _all_ numbers in the string, hence in particular
    # for n0 = 2.0 n1 = 3.0 we are getting list_num = [0.0, 2.0, 1.0, 3.0],
    # so that below we are filtering the odd occurrences.
    assert len(list_num) % 2 == 0
    list_num = [list_num[2 * i + 1] for i in range(int(len(list_num) / 2))]

    # dsl_code is a list of strings; since all DSL programs are single liners,
    # we need to guess the correct line. For now we use the same location as in
    # in the ground truth examples, that is the first line.
    list_op = compute_ops(dsl_code[0])

    try:
        results = compute_result(list_op, list_num)[-1]
    except:  # pylint: disable=bare-except
        results = None
    return results


def is_number(s):
    try:
        float(s)
        return True
    except:  # pylint: disable=bare-except
        return False


def execute_mathqa_program(problem, program):
    """Executes the DSL code for a given problem.

    Args:
      problem: problem formulation (not needed, but we want the same API as
        in the DSL case).
      program: Python code.

    Returns:
      the result of executing of the Python code.
    """
    del problem  # problem only needed in the DSL version.
    # Programs are lists of strings. We need to concatenate them in order to exec.
    program = "\n".join(program)
    var_dict = {}
    try:
        # The logic of this is the following: if exec with timeout is working
        # without exceptions, then we can call exec again and gather the variables.
        exec(program, globals(), var_dict)  # pylint: disable=exec-used
        if "answer" in var_dict and is_number(var_dict["answer"]):
            return float(var_dict["answer"])
        else:
            return None
    except:  # pylint: disable=bare-except
        return None


@gin.configurable(module="trax.data")
def CreateMathQAInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    test=False,
    challenge=False,
    tolerance=0.01,
    cumulative=True,
    python_code=False,
    full_dict=False,
    partial_results=True,
    nlp_rationale=False,
    correct_answer=False,
    answer_in_mathqa_format=True,
    correct_answer_given_reasoning=False,
    category=False,
    order_prediction=False,
    reduced_operation_name=True,
    qed=False,
):
    """Prepares MathQA inputs.

    The generation procedure leaves a lot parameters to be set by the user.
    Currently we support only correct examples in the following sense:
    python execution agrees with the declared answer up to 1%.

    According to this criterion wrong examples such as
    problem: calculate 85184 ÷ ? = 352
    operations ['multiply(n0,n1)']
    are ignored (this should be divide(n0,n1) in this case).

    Args:
      dataset_path: a path with the MathQA dataset.
      train: if True, then generate training examples; if train, test and
        challenge are set to False generate validation examples.
      test: if train is set to False and test is set to True,
        then generate test examples.
      challenge: if train and test are set to False and challenge is set to True,
        then generate challenge examples.
      tolerance: if for a given example relative difference between Python result
        and the result declared in the dataset exceeds the level, then the example
        is dropped; tolerances ranging from 0.1 to 0.001 yield from 18K to 21K
        examples.
      cumulative: if set to True, then generate examples in the format input -
        problem + numbers + op1 + op2 + op3 target - op4 If set to False, then
        examples are in the format input - problem + numbers target - all
        operations.
      python_code: if set to True, then generates python code instead of
        MathQA commands.
      full_dict: if set to True, then Python examples are returned together with
        the DSL code and the NLP rationale.
      partial_results: if set to True, then partial results will be reported as
        part of the input, e.g. input - problem + numbers + op1 + #1 + op2 + #2 +
        op3 + #3, target - op4, where #k is the partial results from operation
        opk. Activated only in cumulative set to True.
      nlp_rationale: if set to True, then input is the problem and the target is
        the nlp rationale.
      correct_answer: if set to True, then input is the problem plus all possible
        answers and the target is the correct answer.
      answer_in_mathqa_format: if set to True, then convert numerical answer to
        the MathQA format and wrap it in the subtract operation.
        E.g. "3.13" is converted to "subtract(const_3_13,const_0)".
      correct_answer_given_reasoning: if set to True, then input is the problem
        plus linear formula plus all possible answers and the target is the
        correct answer.
      category: if set to True, then input is the problem and the target is its
        category.
      order_prediction: if set to True, then input is the problem and a list of
        all operations; with probability 0.5 two operations are swapped; the task
        consists in detecting whether the operations were swapped. See the
        order prediction task in CreateAquaInputs in this file.
      reduced_operation_name: If set to True, then in order prediction consider
        only the operation token without parameterers.
      qed: if set to True, then the reasoning is finished with an additional
        operation qed.

    Returns:
      mathqa_yield_examples: a generator of MathQA examples; the generator yields
      non-tokenized examples - they can be further processed using for example
      the tokenize function from this module
    """
    if train:
        dataset_path = os.path.join(dataset_path, "train.json")
    elif test:
        dataset_path = os.path.join(dataset_path, "test.json")
    elif challenge:
        dataset_path = os.path.join(dataset_path, "challenge_test.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev.json")
    # Opening with GFile allows to use remotely stored files, e.g.
    # in a gs bucket.
    dataset_handle = tf.io.gfile.GFile(dataset_path, "r")
    dataset = json.load(dataset_handle)

    def mathqa_yield_examples(generator=None):
        del generator
        while True:
            for example in itertools.cycle(dataset):
                result = process_single_mathqa_example(example)
                # TODO(henrykm): Remove the first two ifs.
                if not result:
                    continue
                answer_num, python_result, python_program, list_op, list_num = result
                if not answer_num or not python_result[-1]:
                    continue
                if qed:
                    list_op.append("qed")
                if math.isclose(answer_num, python_result[-1], rel_tol=tolerance):
                    input_prefix = example["Problem"]
                    for i in range(len(list_num)):
                        input_prefix += " n{} = {}".format(i, list_num[i])
                    if cumulative:
                        for i in range(len(list_op)):
                            input_values = input_prefix
                            target_values = list_op[i]
                            input_prefix += " " + list_op[i]
                            if partial_results:
                                input_prefix += " #{} = {}".format(i, answer_num)
                            yield (
                                input_values,
                                target_values,
                                np.array([1] * len(target_values)),
                            )
                    elif python_code:
                        input_values = "# " + input_prefix
                        target_values = ""
                        for command in python_program:
                            if "math" in command:
                                target_values += "import math\n"
                                break
                        for command in python_program:
                            if "scipy" in command:
                                target_values += "import scipy\n"
                                break
                        for i in range(len(list_num)):
                            target_values += "n{} = {}\n".format(i, list_num[i])
                        target_values += "\n".join(python_program[:-1])
                        final_line = python_program[-1].split("=")[1]
                        target_values += "\nanswer ={}".format(final_line)
                        var_dict = {}
                        # We generate a python code and want to check whether the answer
                        # is coorect.
                        exec(target_values, globals(), var_dict)  # pylint: disable=exec-used
                        if math.isclose(
                            answer_num, var_dict["answer"], rel_tol=tolerance
                        ):
                            if full_dict:
                                yield (
                                    input_values,
                                    target_values,
                                    example["linear_formula"],
                                    example["Rationale"],
                                )
                            else:
                                yield (
                                    input_values,
                                    target_values,
                                    np.array([1] * len(target_values)),
                                )
                    elif nlp_rationale:
                        input_values = "infer full rationale: " + input_prefix
                        target_values = example["Rationale"]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                    elif correct_answer:
                        input_values = "infer correct answer: " + input_prefix
                        input_values += " " + example["options"]
                        if answer_in_mathqa_format:
                            target_values = str(answer_num)
                            target_values = convert_to_subtract(
                                convert_float_to_mathqa(target_values)
                            )
                        else:
                            target_values = example["correct"]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                    elif correct_answer_given_reasoning:
                        input_values = (
                            "infer correct answer given reasoning: " + input_prefix
                        )
                        input_values += (
                            " " + " ".join(list_op) + " " + example["options"]
                        )
                        target_values = example["correct"]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                    elif category:
                        input_values = "infer category: " + input_prefix
                        target_values = example["category"]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                    elif order_prediction:
                        if np.random.uniform() < 0.5 and len(list_op) >= 2:
                            idx = range(len(list_op))
                            i1, i2 = random.sample(idx, 2)
                            list_op[i1], list_op[i2] = list_op[i2], list_op[i1]
                            target_values = "not_ordered"
                        else:
                            target_values = "ordered"
                        if reduced_operation_name:
                            list_op = [op.split("(")[0] for op in list_op]
                        input_values = (
                            "order prediction: "
                            + input_prefix
                            + " "
                            + " ".join(list_op)
                        )
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                    else:
                        input_values = "infer full calculation: " + input_prefix
                        target_values = example["linear_formula"]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )

    return mathqa_yield_examples
