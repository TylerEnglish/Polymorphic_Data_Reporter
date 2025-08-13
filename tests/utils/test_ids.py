from src.utils.ids import slugify, stable_hash, short_id, make_dataset_slug, make_chart_id

def test_slugify_basic_and_separators():
    assert slugify("Hello World!") == "hello-world"
    assert slugify("a/b\\c  d") == "a-b-c-d"

def test_stable_hash_deterministic_and_json_order_insensitive():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert stable_hash(a) == stable_hash(b)

def test_short_id_length_and_prefix_changes():
    x = {"x": 1}
    y = {"x": 2}
    sx = short_id(x, 8)
    sy = short_id(y, 8)
    assert len(sx) == 8
    assert len(sy) == 8
    assert sx != sy

def test_make_dataset_slug_path_like():
    s = make_dataset_slug(r"file://data\raw\sales_demo")
    # slugify uses '-' separators
    assert "sales-demo" in s
    assert s.startswith("file-data-raw")

def test_make_chart_id_stable_for_same_payload():
    topic_key = {"dataset_slug": "demo", "fields": ["a","b"]}
    c1 = make_chart_id(topic_key, "bar")
    c2 = make_chart_id(topic_key, "bar")
    assert c1 == c2
    c3 = make_chart_id(topic_key, "line")
    assert c3 != c1