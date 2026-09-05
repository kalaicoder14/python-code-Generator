Python Code Generator

An AI-powered Python code generation and debugging application built with **Streamlit**, **Hugging Face Transformers**, and the **Qwen2.5-Coder** model.

The application allows users to select a Python programming problem or enter their own problem. The AI generates Python code, checks its syntax, executes the code with user input, detects runtime errors, and can automatically fix the generated code.



Features

- 🤖 AI-powered Python code generation
- 📝 Predefined Python programming problems
- ✍️ Custom problem input
- 🔢 Multiple methods for solving the same problem
- ✅ Automatic Python syntax checking
- ▶️ Execute generated Python code
- ⚠️ Runtime error detection
- 🔧 Automatic runtime error fixing
- 🔄 Generate another solution method
- 📋 Copy generated code
- 📥 Download generated code
- 🎨 Light-blue user interface
- 💻 Simple and interactive Streamlit interface



 AI Model

This project uses:

**Qwen/Qwen2.5-Coder-0.5B-Instruct**

The model is loaded using Hugging Face Transformers and PyTorch.



 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web application interface |
| PyTorch | AI model execution |
| Hugging Face Transformers | Loading and running the AI model |
| Qwen2.5-Coder | Python code generation |
| AST | Python syntax validation |
| Subprocess | Executing generated Python code |



 📂 Project Structure


AI-Python-Code-Generator/
│
├── app.py
├── requirements.txt
└── README.md
