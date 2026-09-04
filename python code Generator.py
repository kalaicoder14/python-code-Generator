
import streamlit as st
import torch
import ast
import subprocess
import tempfile
import os
import sys
import re
import html

from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Python Code Generator",
    layout="wide"
)



# =========================================================
# UI BACKGROUND THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #eaf6ff 0%,
            #f7fbff 45%,
            #eef7ff 100%
        );
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        text-align: center;
        color: #123b5d;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    h2 {
        color: #174a70;
        font-weight: 700;
    }

    h3 {
        color: #205b82;
        font-weight: 650;
    }

    p, label, span {
        color: #243b53;
    }

    div[data-baseweb="select"] > div {
        background-color: black;
        border-radius: 12px;
        border: 1px solid #b8d7ed;
    }

    textarea {
        background-color: #ffffff !important;
        color: #172b4d !important;
        border-radius: 12px !important;
        border: 1px solid #b8d7ed !important;
    }

    textarea:focus {
        border: 2px solid #4a9ed6 !important;
        box-shadow: 0 0 8px rgba(74, 158, 214, 0.25) !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #1976b9, #329bd2);
        color: white;
        font-size: 16px;
        font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(25, 118, 185, 0.25);
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 12px;
        background-color: white;
        color: #17699c;
        border: 2px solid #5ca8d6;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        background-color: #edf8ff;
    }

    div[data-testid="stCodeBlock"] {
        border-radius: 14px;
        border: 1px solid #b8d7ed;
        box-shadow: 0 5px 18px rgba(40, 90, 120, 0.08);
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border: none;
        border-top: 1px solid #c9e0ef;
        margin: 25px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


# =========================================================
# CLEAN GENERATED CODE
# =========================================================

def clean_generated_code(code):

    code = code.strip()

    # Remove markdown
    code = re.sub(
        r"```python\s*",
        "",
        code,
        flags=re.IGNORECASE
    )

    code = re.sub(
        r"```\s*",
        "",
        code
    )

    lines = code.splitlines()

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        # Remove AI explanations
        if stripped.startswith(
            (
                "This program",
                "This Python program",
                "This code",
                "Explanation:",
                "Explanation",
                "The above program",
                "The program",
                "Here is",
                "Here’s"
            )
        ):
            break

        # Remove LaTeX explanation
        if "\\(" in line or "\\)" in line:
            break

        cleaned_lines.append(line)

    return "\n".join(
        cleaned_lines
    ).strip()


# =========================================================
# SYNTAX CHECK
# =========================================================

def check_syntax(code):

    try:

        ast.parse(code)

        return True, "Valid Python code"

    except SyntaxError as e:

        line_number = e.lineno if e.lineno else "unknown"

        return False, (
            f"Line {line_number}: {e.msg}"
        )


# =========================================================
# GENERATE CODE
# =========================================================

def generate_code(
    problem,
    method_number,
    previous_methods
):

    if method_number == 1:

        method_instruction = """
Use the simplest beginner-friendly approach.

Use basic Python syntax.
"""

    else:

        method_instruction = """
Generate a DIFFERENT solution from all previous methods.

Do NOT copy previous solutions.

Use a different programming technique where possible.

Possible techniques:

- function
- loop
- while loop
- built-in functions
- recursion
- list comprehension
- class
"""


    previous_text = ""

    if previous_methods:

        previous_text = """

Previous solutions:

""" + "\n\n".join(
            previous_methods
        ) + """

IMPORTANT:
Do NOT copy these solutions.
Create a different implementation.
"""


    system_prompt = """
You are an expert Python programmer.

Generate a complete Python 3 program.

STRICT RULES:

1. Return ONLY Python code.
2. Do NOT write explanations.
3. Do NOT write Markdown.
4. Do NOT use code fences.
5. Do NOT use LaTeX.
6. Do NOT write "This program..."
7. Do NOT explain the code.
8. The code must be complete.
9. The code must be syntactically valid.
10. Close every bracket.
11. Close every parenthesis.
12. Close every quote.
13. Close every f-string.
14. Use correct programming logic.
15. Use input() when user input is required.
16. Do NOT put prompts inside input().
17. Use input() or input().strip().
18. Print only the required final result.
19. Do NOT print unnecessary messages.
20. Do NOT print the user's input.
21. Do not stop in the middle of a line.
22. The program must be executable directly with Python.
23. Check the code before returning it.
"""


    user_prompt = f"""
Problem:

{problem}

Method number:

{method_number}

{method_instruction}

{previous_text}

Generate ONLY the complete Python program.

No explanation.
No Markdown.
No code fences.
No text after the program.
"""


    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": user_prompt
        }

    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        text,
        return_tensors="pt"
    )


    outputs = model.generate(

        **inputs,

        max_new_tokens=500,

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id

    )


    generated_ids = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    code = tokenizer.decode(

        generated_ids,

        skip_special_tokens=True

    )


    code = clean_generated_code(code)

    return code


# =========================================================
# AUTOMATIC CODE FIX
# =========================================================

def repair_code(code, problem):

    repair_prompt = f"""
You are a Python code debugger.

The following Python code has a syntax or formatting problem.

Problem:
{problem}

Code:
{code}

Fix the code.

STRICT RULES:

1. Return ONLY corrected Python code.
2. Do NOT explain anything.
3. Do NOT use Markdown.
4. Do NOT use code fences.
5. Do NOT use LaTeX.
6. Keep the original purpose.
7. Make the code complete.
8. Make the code syntactically valid.
9. Close all brackets.
10. Close all parentheses.
11. Close all quotes.
12. Close all f-strings.
13. Use input() without prompts.
14. Print only the final result.
15. Return the complete executable Python program.
"""


    messages = [

        {
            "role": "system",
            "content": (
                "You are an expert Python debugger. "
                "Return only valid Python code."
            )
        },

        {
            "role": "user",
            "content": repair_prompt
        }

    ]


    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True

    )


    inputs = tokenizer(

        text,

        return_tensors="pt"

    )


    outputs = model.generate(

        **inputs,

        max_new_tokens=500,

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id

    )


    generated_ids = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    fixed_code = tokenizer.decode(

        generated_ids,

        skip_special_tokens=True

    )


    fixed_code = clean_generated_code(
        fixed_code
    )


    return fixed_code



# =========================================================
# FIX RUNTIME ERROR AUTOMATICALLY
# =========================================================

def fix_runtime_error(code, problem, error_message):

    system_prompt = """
You are an expert Python debugging assistant.

A Python program produced a runtime error.
Fix the actual runtime error.

STRICT RULES:
1. Return ONLY valid Python code.
2. Do NOT use Markdown.
3. Do NOT use code fences.
4. Do NOT explain anything.
5. Keep the original purpose of the program.
6. Do not unnecessarily rewrite working code.
7. Make sure all variables are defined.
8. Make sure input handling is correct.
9. Make sure the program is complete.
10. Return only complete executable Python 3 code.
"""

    user_prompt = f"""
Original Problem:
{problem}

Current Python Code:
{code}

Runtime Error:
{error_message}

Return ONLY the corrected Python code.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    fixed_code = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return clean_generated_code(fixed_code)

# =========================================================
# GENERATE + AUTO FIX
# =========================================================

def generate_valid_code(
    problem,
    method_number,
    previous_methods
):

    # First generation

    code = generate_code(

        problem,

        method_number,

        previous_methods

    )


    valid, error = check_syntax(code)


    if valid:

        return code, True, ""


    # Automatically repair once

    fixed_code = repair_code(

        code,

        problem

    )


    valid, error = check_syntax(
        fixed_code
    )


    if valid:

        return fixed_code, True, ""


    return fixed_code, False, error


# =========================================================
# CLEAN PROGRAM OUTPUT
# =========================================================

def clean_output(
    output,
    user_input
):

    if not output:

        return ""


    # Remove common prompts

    output = re.sub(

        r"(Enter|Input|Please enter)"
        r"[^:\n]*:\s*",

        "",

        output,

        flags=re.IGNORECASE

    )


    # Remove echoed input values

    if user_input.strip():

        input_values = (
            user_input
            .strip()
            .splitlines()
        )

        for value in input_values:

            value = value.strip()

            if value:

                output = output.replace(
                    value,
                    "",
                    1
                )


    # Remove extra blank lines

    output = re.sub(
        r"\n\s*\n+",
        "\n",
        output
    )


    return output.strip()


# =========================================================
# EXECUTE CODE
# =========================================================

def execute_code(
    code,
    user_input
):

    filename = None

    try:

        valid, error = check_syntax(
            code
        )

        if not valid:

            return "", (
                "Syntax Error: "
                + error
            )


        # Temporary file

        with tempfile.NamedTemporaryFile(

            mode="w",

            suffix=".py",

            delete=False,

            encoding="utf-8"

        ) as file:

            file.write(code)

            filename = file.name


        # Run using current Python

        result = subprocess.run(

            [
                sys.executable,
                filename
            ],

            input=user_input,

            capture_output=True,

            text=True,

            timeout=5

        )


        output = result.stdout

        error = result.stderr


        output = clean_output(
            output,
            user_input
        )


        return output, error


    except subprocess.TimeoutExpired:

        return "", (
            "Program took too long to execute."
        )


    except Exception as e:

        return "", str(e)


    finally:

        if filename:

            if os.path.exists(filename):

                try:

                    os.remove(filename)

                except:

                    pass


# =========================================================
# COPY BUTTON
# =========================================================

def copy_button(code):

    escaped_code = html.escape(
        code
    )

    html_code = f"""
    <div style="margin-bottom:15px;">

        <textarea
            id="code_box"
            style="
                position:absolute;
                left:-9999px;
            "
        >{escaped_code}</textarea>

        <button
            onclick="
                const code = document.getElementById('code_box').value;

                navigator.clipboard.writeText(code);

                this.innerText = '✅ Copied!';

                setTimeout(() => {{
                    this.innerText = '📋 Copy Code';
                }}, 2000);
            "
            style="
                padding:10px 20px;
                border:none;
                border-radius:8px;
                cursor:pointer;
                font-size:16px;
                font-weight:bold;
            "
        >
            📋 Copy Code
        </button>

    </div>
    """

    st.components.v1.html(
        html_code,
        height=55
    )


# =========================================================
# SESSION STATE
# =========================================================

if "methods" not in st.session_state:

    st.session_state.methods = []


if "current_code" not in st.session_state:

    st.session_state.current_code = ""


if "method_number" not in st.session_state:

    st.session_state.method_number = 1


if "last_problem" not in st.session_state:

    st.session_state.last_problem = ""


if "last_error" not in st.session_state:

    st.session_state.last_error = ""


if "show_fix_button" not in st.session_state:

    st.session_state.show_fix_button = False


if "code_fixed" not in st.session_state:

    st.session_state.code_fixed = False


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="font-size:42px;margin-bottom:5px;color:#123b5d;">
            python Code Generator
        </h1>
        <p style="font-size:18px;color:#52748c;margin-top:0;">
            Generate Code • Run 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

with st.spinner(
    "Loading AI model... Please wait..."
):

    tokenizer, model = load_model()


#st.success(
   



# =========================================================
# PROBLEM LIST
# =========================================================

st.subheader(
    "Select a Python Problem"
)


problems = {

    "Simple Interest":
        "Calculate simple interest using principal, rate and time",

    "Prime Number":
        "Check whether a number is prime or not",

    "Factorial":
        "Find the factorial of a given number",

    "Reverse String":
        "Reverse a given string",

    "Even or Odd":
        "Check whether a number is even or odd",

    "Fibonacci Series":
        "Generate Fibonacci series for n terms",

    "Largest Number":
        "Find the largest number in a list",

    "Palindrome":
        "Check whether a string is palindrome or not",

    "Student Average and Grade":
        "Calculate student average marks and display grade",

    "Leap Year":
        "Check whether a year is a leap year or not",

    "Sum of Digits":
        "Calculate the sum of digits of a number",

    "Armstrong Number":
        "Check whether a number is an Armstrong number or not"

}


selected_problem = st.selectbox(

    "Choose a problem",

    list(problems.keys())

)


# =========================================================
# CUSTOM PROBLEM
# =========================================================

st.subheader(
    " Enter Your Own Problem"
)


custom_problem = st.text_area(

    "Describe what Python program you want:",

    placeholder=(
        "Example: "
        "Create a Python program to calculate "
        "the area of a circle"
    ),

    height=100

)


if custom_problem.strip():

    problem = custom_problem.strip()

else:

    problem = problems[
        selected_problem
    ]


st.info(
    "problem: " + problem
)


# =========================================================
# RESET WHEN PROBLEM CHANGES
# =========================================================

if (
    st.session_state.last_problem
    != problem
):

    st.session_state.methods = []

    st.session_state.current_code = ""

    st.session_state.method_number = 1

    st.session_state.last_problem = problem
    st.session_state.last_error = ""
    st.session_state.show_fix_button = False
    st.session_state.code_fixed = False


# =========================================================
# GENERATE CODE
# =========================================================

if st.button(

    "Generate Python Code",

    type="primary",

    use_container_width=True

):

    with st.spinner(
        "Generating and checking code..."
    ):

        code, valid, error = (
            generate_valid_code(

                problem,

                st.session_state.method_number,

                st.session_state.methods

            )
        )


    if valid:

        st.session_state.current_code = code

        st.session_state.methods.append(
            code
        )

        st.session_state.last_error = ""
        st.session_state.show_fix_button = False
        st.session_state.code_fixed = False

        st.success(

            " Method "
            + str(
                st.session_state.method_number
            )
            + " generated successfully!"

        )

    else:

        st.error(
            "❌ Unable to generate valid code."
        )

        st.warning(error)

        st.code(
            code,
            language="python"
        )


# =========================================================
# DISPLAY CODE
# =========================================================

if st.session_state.current_code:

    st.divider()


    st.subheader(

        "Generated Python Code - Method "
        + str(
            st.session_state.method_number
        )

    )


    # Code display

    st.code(

        st.session_state.current_code,

        language="python"

    )


    # Custom copy button

    copy_button(
        st.session_state.current_code
    )


    # Download

    st.download_button(

        label="⬇️ Download Python File",

        data=st.session_state.current_code,

        file_name="generated_code.py",

        mime="text/plain",

        use_container_width=True

    )


    # =====================================================
    # USER INPUT
    # =====================================================

    st.subheader(
        "Enter Program Input"
    )


    user_input = st.text_area(

        "Input",

        placeholder=(
            "Enter each input value "
            "on a separate line.\n\n"
            "Example:\n"
            "1000\n"
            "5\n"
            "2"
        ),

        height=140,

        key="program_input"

    )


    st.caption(
        "Enter multiple values on separate lines."
    )


    # =====================================================
    # RUN
    # =====================================================

    if st.button(

        "▶️ Run Python Code",

        type="primary",

        use_container_width=True

    ):

        with st.spinner(
            "Running Python program..."
        ):

            output, error = execute_code(

                st.session_state.current_code,

                user_input

            )


        # Output

        if output:

            st.subheader(
                "🖥️ Program Output"
            )

            st.code(
                output,
                language="text"
            )


        # Error

        if error:

            st.subheader(
                "❌ Program Error"
            )

            st.code(
                error,
                language="text"
            )


        # Success without output

        if not output and not error:

            st.success(
                "✅ Program executed successfully."
            )


    # =====================================================
    # ANOTHER METHOD
    # =====================================================

    st.divider()


    if st.button(

        "🔄 Generate Another Method",

        use_container_width=True

    ):

        new_method_number = (
            st.session_state.method_number + 1
        )


        with st.spinner(
            "Generating and checking another solution..."
        ):

            code, valid, error = (
                generate_valid_code(

                    problem,

                    new_method_number,

                    st.session_state.methods

                )
            )


        if valid:

            st.session_state.method_number = (
                new_method_number
            )

            st.session_state.current_code = code

            st.session_state.methods.append(
                code
            )

            st.session_state.last_error = ""
            st.session_state.show_fix_button = False
            st.session_state.code_fixed = False

            st.success(

                "✅ Method "
                + str(
                    new_method_number
                )
                + " generated!"

            )

            st.rerun()


        else:

            st.error(
                "❌ Could not generate another valid method."
            )

            st.warning(error)

            st.code(
                code,
                language="python"
            )


