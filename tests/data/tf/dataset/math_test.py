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

"""Tests for trax.data.tf_inputs."""


import gin
import tensorflow as tf

from trax.data.preprocessing.inputs import batcher  # noqa: F401
from trax.data.preprocessing.tf import math as dataset_math


class TFDatasetMathTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_process_single_mathqa_example_0(self):
        # This is the first problem in the MathQA dataset.
        example = {
            "Problem": "the banker ' s gain of a certain sum due 3 years hence at 10 % "
            "per annum is rs . 36 . what is the present worth ?",
            "Rationale": '"explanation : t = 3 years r = 10 % td = ( bg × 100 ) / tr = ( '
            "36 × 100 ) / ( 3 × 10 ) = 12 × 10 = rs . 120 td = ( pw × tr )"
            " / 100 ⇒ 120 = ( pw × 3 × 10 ) / 100 ⇒ 1200 = pw × 3 pw = "
            '1200 / 3 = rs . 400 answer : option a"',
            "options": "a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) "
            "none of these",
            "correct": "a",
            "annotated_formula": "divide(multiply(const_100, divide(multiply(36, const_100), "
            "multiply(3, 10))), multiply(3, 10))",
            "linear_formula": "multiply(n2,const_100)|multiply(n0,n1)|divide(#0,#1)|multiply(#2,const_100)|divide(#3,#1)|",
            "category": "gain",
        }

        (
            answer_num,
            python_result,
            python_program,
            list_op,
            list_num,
        ) = dataset_math.process_single_mathqa_example(example)

        self.assertEqual(answer_num, 400)  # we know it, because correct answer is a)
        self.assertEqual(python_result, [3600.0, 30.0, 120.0, 12000.0, 400.0])

        self.assertEqual(
            python_program,
            [
                "t0 = n2 * 100.0",
                "t1 = n0 * n1",
                "t2 = t0 / t1",
                "t3 = t2 * 100.0",
                "t4 = t3 / t1",
            ],
        )
        self.assertEqual(
            list_op,
            [
                "multiply(n2,const_100)",
                "multiply(n0,n1)",
                "divide(#0,#1)",
                "multiply(#2,const_100)",
                "divide(#3,#1)",
            ],
        )
        self.assertEqual(list_num, [3.0, 10.0, 36.0])

    def test_process_single_mathqa_example_1(self):
        # This is the third problem in the MathQA dataset.
        example = {
            "Problem": "sophia finished 2 / 3 of a book . she calculated that she "
            "finished 90 more pages than she has yet to read . how long is her"
            " book ?",
            "Rationale": "let xx be the total number of pages in the book , then she "
            "finished 23 ⋅ x 23 ⋅ x pages . then she has x − 23 ⋅ x = "
            "13 ⋅ xx − 23 ⋅ x = 13 ⋅ x pages left . 23 ⋅ x − 13 "
            "⋅ x = 9023 ⋅ x − 13 ⋅ x = 90 13 ⋅ x = 9013 ⋅ x = 90 x"
            " = 270 x = 270 so the book is 270 pages long . answer : b",
            "options": "a ) 229 , b ) 270 , c ) 877 , d ) 266 , e ) 281",
            "correct": "b",
            "annotated_formula": "divide(90, subtract(const_1, divide(2, 3)))",
            "linear_formula": "divide(n0,n1)|subtract(const_1,#0)|divide(n2,#1)",
            "category": "general",
        }

        (
            answer_num,
            python_result,
            python_program,
            list_op,
            list_num,
        ) = dataset_math.process_single_mathqa_example(example)

        self.assertEqual(answer_num, 270)  # we know it, because correct answer is b)
        self.assertAllClose(
            python_result, [0.6666666666666666, 0.33333333333333337, 269.99999999999994]
        )
        self.assertEqual(
            python_program, ["t0 = n0 / n1", "t1 = 1.0 - t0", "t2 = n2 / t1"]
        )
        self.assertEqual(
            list_op, ["divide(n0,n1)", "subtract(const_1,#0)", "divide(n2,#1)"]
        )
        self.assertEqual(list_num, [2.0, 3.0, 90.0])

    def test_process_single_mathqa_example_with_import(self):
        # This is a training MathQA problem which involve an import.
        example = {
            "Problem": "the length of a rectangular garden is three times its width . if "
            "the area of the rectangular garden is 588 square meters , then "
            "what is the width of the rectangular garden ?",
            "Rationale": '"let x be the width of the garden . 3 x ^ 2 = 588 x ^ 2 = 196 x '
            '= 14 the answer is c ."',
            "options": "a ) 12 , b ) 13 , c ) 14 , d ) 15 , e ) 16",
            "correct": "c",
            "annotated_formula": "sqrt(divide(588, const_3))",
            "linear_formula": "divide(n0,const_3)|sqrt(#0)|",
            "category": "geometry",
        }

        (
            answer_num,
            python_result,
            python_program,
            list_op,
            list_num,
        ) = dataset_math.process_single_mathqa_example(example)

        self.assertEqual(answer_num, 14)  # we know it, because correct answer is c)
        self.assertAllClose(python_result, [196, 14])
        self.assertEqual(
            python_program, ["t0 = n0 / 3.0", "t1 = math.sqrt(max(0, t0))"]
        )
        self.assertEqual(list_op, ["divide(n0,const_3)", "sqrt(#0)"])
        self.assertEqual(list_num, [588])

        # Below we execute twice the Python program and once the DSL program.
        target_values = "import math\n"
        problem = example["Problem"]
        for i in range(len(list_num)):
            target_values += "n{} = {}\n".format(i, list_num[i])
            problem += " n{} = {}".format(i, list_num[i])
        target_values += "\n".join(python_program[:-1])
        final_line = python_program[-1].split("=")[1]
        target_values += "\nanswer ={}".format(final_line)
        var_dict = {}
        exec(target_values, globals(), var_dict)  # pylint: disable=exec-used
        self.assertAllClose(var_dict["answer"], 14)
        self.assertAllClose(
            dataset_math.execute_mathqa_program(problem, target_values.split("\n")), 14
        )
        self.assertAllClose(
            dataset_math.execute_mathqa_dsl_program(
                problem, [example["linear_formula"]]
            ),
            14,
        )


if __name__ == "__main__":
    tf.test.main()
