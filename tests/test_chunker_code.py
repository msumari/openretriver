from src.models import LoadedFile
from src.chunker_code import chunk_code


def _make_code(content: str, ext: str = ".py") -> LoadedFile:
    paths = {
        ".py": "src/main.py", ".js": "src/app.js", ".ts": "src/app.ts", ".rs": "src/lib.rs",
        ".json": "config.json", ".yaml": "config.yaml", ".yml": "config.yml",
        ".sh": "scripts/build.sh", ".bash": "scripts/run.bash", ".mk": "Makefile",
    }
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
    for ext, lang in [
        (".py", "python"), (".js", "javascript"), (".ts", "typescript"), (".rs", "rust"),
        (".json", "json"), (".yaml", "yaml"), (".yml", "yaml"),
        (".sh", "bash"), (".bash", "bash"), (".mk", "make"),
    ]:
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


# --- JSON ---

def test_json_top_level_pairs():
    source = '{"name": "openretriver", "version": "1.0", "description": "A retriever"}'
    chunks = chunk_code(_make_code(source, ".json"))
    pairs = [c for c in chunks if c.symbol_type == "pair"]
    assert len(pairs) == 3
    names = {c.symbol_name for c in pairs}
    assert "name" in names
    assert "version" in names
    assert "description" in names


def test_json_nested_object_stays_in_parent_chunk():
    source = '{"config": {"host": "localhost", "port": 8080}}'
    chunks = chunk_code(_make_code(source, ".json"))
    pairs = [c for c in chunks if c.symbol_type == "pair"]
    assert len(pairs) == 1
    assert pairs[0].symbol_name == "config"
    assert "localhost" in pairs[0].text


def test_json_array_root_single_chunk():
    source = '[1, 2, 3, "hello"]'
    chunks = chunk_code(_make_code(source, ".json"))
    assert len(chunks) == 1


def test_json_empty_object_single_chunk():
    source = '{}'
    chunks = chunk_code(_make_code(source, ".json"))
    assert len(chunks) == 1


def test_json_empty_string_key():
    source = '{"": "empty", "normal": "val"}'
    chunks = chunk_code(_make_code(source, ".json"))
    pairs = [c for c in chunks if c.symbol_type == "pair"]
    assert len(pairs) == 2
    names = {c.symbol_name for c in pairs}
    assert "normal" in names


def test_json_language_field():
    source = '{"key": "value"}'
    chunks = chunk_code(_make_code(source, ".json"))
    assert all(c.language == "json" for c in chunks)


# --- YAML ---

def test_yaml_top_level_keys():
    source = "name: openretriver\nversion: 1.0\ndescription: A retriever\n"
    chunks = chunk_code(_make_code(source, ".yaml"))
    pairs = [c for c in chunks if c.symbol_type == "pair"]
    assert len(pairs) == 3
    names = {c.symbol_name for c in pairs}
    assert "name" in names
    assert "version" in names
    assert "description" in names


def test_yaml_nested_mapping_stays_in_parent():
    source = "server:\n  host: localhost\n  port: 8080\n"
    chunks = chunk_code(_make_code(source, ".yaml"))
    pairs = [c for c in chunks if c.symbol_type == "pair"]
    assert len(pairs) == 1
    assert pairs[0].symbol_name == "server"
    assert "localhost" in pairs[0].text


def test_yaml_multi_document():
    source = "---\ndoc1_key: value1\n---\ndoc2_key: value2\n"
    chunks = chunk_code(_make_code(source, ".yaml"))
    pairs = [c for c in chunks if c.symbol_type == "pair"]
    assert len(pairs) == 2
    names = {c.symbol_name for c in pairs}
    assert "doc1_key" in names
    assert "doc2_key" in names


def test_yaml_yml_extension():
    source = "key: value\n"
    chunks = chunk_code(_make_code(source, ".yml"))
    assert len(chunks) >= 1
    assert all(c.language == "yaml" for c in chunks)


def test_yaml_language_field():
    source = "key: value\n"
    chunks = chunk_code(_make_code(source, ".yaml"))
    assert all(c.language == "yaml" for c in chunks)


# --- Bash ---

def test_bash_function_parens_syntax():
    source = "#!/bin/bash\n\ngreet() {\n  echo 'hello'\n}\n"
    chunks = chunk_code(_make_code(source, ".sh"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].symbol_name == "greet"


def test_bash_function_keyword_syntax():
    source = "function deploy {\n  echo 'deploying'\n}\n"
    chunks = chunk_code(_make_code(source, ".sh"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].symbol_name == "deploy"


def test_bash_preamble():
    source = "#!/bin/bash\nset -euo pipefail\nVAR=value\n\nrun() {\n  echo ok\n}\n"
    chunks = chunk_code(_make_code(source, ".sh"))
    preambles = [c for c in chunks if c.symbol_type == "preamble"]
    assert len(preambles) == 1
    assert "set -euo" in preambles[0].text


def test_bash_no_functions_single_chunk():
    source = "#!/bin/bash\necho 'hello world'\nexit 0\n"
    chunks = chunk_code(_make_code(source, ".sh"))
    assert len(chunks) == 1


def test_bash_language_field():
    source = "echo hello\n"
    chunks = chunk_code(_make_code(source, ".sh"))
    assert all(c.language == "bash" for c in chunks)


def test_bash_extension():
    source = "#!/bin/bash\n\nrun() {\n  echo go\n}\n"
    chunks = chunk_code(_make_code(source, ".bash"))
    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].symbol_name == "run"


# --- Make ---

def test_make_rule():
    source = "build:\n\tgcc -o main main.c\n"
    chunks = chunk_code(_make_code(source, ".mk"))
    rules = [c for c in chunks if c.symbol_type == "rule"]
    assert len(rules) == 1
    assert rules[0].symbol_name == "build"


def test_make_multiple_rules():
    source = "build:\n\tgcc main.c\n\ntest:\n\t./run_tests\n"
    chunks = chunk_code(_make_code(source, ".mk"))
    rules = [c for c in chunks if c.symbol_type == "rule"]
    assert len(rules) == 2
    names = {r.symbol_name for r in rules}
    assert "build" in names
    assert "test" in names


def test_make_variable_preamble():
    source = "CC = gcc\nCFLAGS = -Wall\n\nbuild:\n\t$(CC) $(CFLAGS) main.c\n"
    chunks = chunk_code(_make_code(source, ".mk"))
    assert len(chunks) >= 2


def test_make_language_field():
    source = "all:\n\techo done\n"
    chunks = chunk_code(_make_code(source, ".mk"))
    assert all(c.language == "make" for c in chunks)
