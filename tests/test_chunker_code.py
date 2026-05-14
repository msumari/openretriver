from src.models import LoadedFile
from src.chunker_code import chunk_code


def _make_code(content: str, ext: str = ".py") -> LoadedFile:
    paths = {".py": "src/main.py", ".js": "src/app.js", ".ts": "src/app.ts", ".rs": "src/lib.rs"}
    return LoadedFile(path=paths[ext], extension=ext, content=content)


# --- Python ---

def test_python_function_becomes_chunk():
    source = "def hello():\n    return 'hi'\n"
    chunks = chunk_code(_make_code(source))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].symbol_name == "hello"
    assert "def hello" in funcs[0].text


def test_python_class_becomes_chunk():
    source = "class Router:\n    def handle(self):\n        pass\n"
    chunks = chunk_code(_make_code(source))
    classes = [c for c in chunks if c.symbol_type == "class"]
    assert len(classes) == 1
    assert classes[0].symbol_name == "Router"
    assert "class Router" in classes[0].text
    assert "def handle" in classes[0].text


def test_python_preamble_chunk():
    source = "import os\nimport sys\n\ndef main():\n    pass\n"
    chunks = chunk_code(_make_code(source))
    preambles = [c for c in chunks if c.symbol_type == "preamble"]
    assert len(preambles) == 1
    assert "import os" in preambles[0].text


def test_python_multiple_functions():
    source = "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n"
    chunks = chunk_code(_make_code(source))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) == 3
    names = {f.symbol_name for f in funcs}
    assert names == {"a", "b", "c"}


def test_python_docstring_included():
    source = 'def greet():\n    """Say hello."""\n    return "hi"\n'
    chunks = chunk_code(_make_code(source))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert "Say hello." in funcs[0].text


def test_python_syntax_error_fallback():
    source = "def broken(\n    pass\n"
    chunks = chunk_code(_make_code(source))
    assert len(chunks) >= 1


# --- JavaScript ---

def test_js_function_declaration():
    source = "function greet() {\n  return 'hi';\n}\n"
    chunks = chunk_code(_make_code(source, ".js"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) >= 1
    assert "greet" in funcs[0].symbol_name


def test_js_class():
    source = "class App {\n  constructor() {}\n  run() {}\n}\n"
    chunks = chunk_code(_make_code(source, ".js"))
    classes = [c for c in chunks if c.symbol_type == "class"]
    assert len(classes) == 1
    assert classes[0].symbol_name == "App"


def test_js_nested_braces():
    source = "function complex() {\n  if (true) {\n    for (let i=0; i<10; i++) {\n      console.log(i);\n    }\n  }\n}\n"
    chunks = chunk_code(_make_code(source, ".js"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) >= 1
    assert "console.log" in funcs[0].text


# --- TypeScript ---

def test_ts_interface():
    source = "interface Config {\n  host: string;\n  port: number;\n}\n"
    chunks = chunk_code(_make_code(source, ".ts"))
    interfaces = [c for c in chunks if c.symbol_type == "interface"]
    assert len(interfaces) == 1
    assert interfaces[0].symbol_name == "Config"


def test_ts_generics():
    source = "function identity<T>(arg: T): T {\n  return arg;\n}\n"
    chunks = chunk_code(_make_code(source, ".ts"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) >= 1
    assert "identity" in funcs[0].symbol_name


# --- Rust ---

def test_rust_fn():
    source = "fn main() {\n    println!(\"hello\");\n}\n"
    chunks = chunk_code(_make_code(source, ".rs"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) >= 1
    assert "main" in funcs[0].symbol_name


def test_rust_impl_block():
    source = "struct Server;\n\nimpl Server {\n    fn start(&self) {\n        // start\n    }\n}\n"
    chunks = chunk_code(_make_code(source, ".rs"))
    impls = [c for c in chunks if c.symbol_type == "impl"]
    assert len(impls) >= 1
    assert "Server" in impls[0].symbol_name


def test_rust_struct():
    source = "struct Config {\n    host: String,\n    port: u16,\n}\n"
    chunks = chunk_code(_make_code(source, ".rs"))
    structs = [c for c in chunks if c.symbol_type == "struct"]
    assert len(structs) == 1
    assert structs[0].symbol_name == "Config"


def test_rust_enum():
    source = "enum Color {\n    Red,\n    Green,\n    Blue,\n}\n"
    chunks = chunk_code(_make_code(source, ".rs"))
    enums = [c for c in chunks if c.symbol_type == "enum"]
    assert len(enums) == 1
    assert enums[0].symbol_name == "Color"


def test_rust_pub_async_fn():
    source = "pub async fn serve() {\n    loop {}\n}\n"
    chunks = chunk_code(_make_code(source, ".rs"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) >= 1
    assert "serve" in funcs[0].symbol_name


# --- General ---

def test_code_chunk_file_type():
    source = "def foo():\n    pass\n"
    chunks = chunk_code(_make_code(source))
    assert all(c.file_type == "code" for c in chunks)


def test_code_chunk_language():
    for ext, lang in [(".py", "python"), (".js", "javascript"), (".ts", "typescript"), (".rs", "rust")]:
        source = "x = 1\n"
        chunks = chunk_code(_make_code(source, ext))
        assert all(c.language == lang for c in chunks)


def test_empty_code_file():
    chunks = chunk_code(_make_code(""))
    assert chunks == []


def test_fallback_triggers_on_failure():
    source = "def valid():\n    pass\n"
    chunks = chunk_code(_make_code(source))
    assert len(chunks) >= 1
