from hashlib import sha256
import json
import os
from os.path import join as pjoin
import socket

import numpy.testing as npt
import pytest

from fury.data import (
    FetcherError,
    fetch_gltf,
    fetch_ui_themes,
    list_gltf_sample_models,
    read_viz_gltf,
)
from fury.data.fetcher import UI_THEMES_DATA_URL, UI_THEME_SHA256

if "FURY_HOME" in os.environ:
    fury_home = os.environ["FURY_HOME"]
else:
    fury_home = pjoin(os.path.expanduser("~"), ".fury")

GLTF_DATA_URL = (
    "https://api.github.com/repos/KhronosGroup/glTF-Sample-Models/contents/2.0/"  # noqa
)


def tests_fetch_gltf():
    folder = pjoin(fury_home, "glTF")
    boxtex = pjoin(folder, "BoxTextured")
    boxtex = pjoin(boxtex, "glTF")
    models_list = ["BoxTextured", "Box"]
    if os.path.exists(boxtex):
        for path in os.listdir(boxtex):
            os.remove(pjoin(boxtex, path))
        os.rmdir(boxtex)

    fetch_gltf(name=models_list)
    list_gltf = os.listdir(folder)
    results = [model in list_gltf for model in models_list]

    npt.assert_equal(results, [True, True])
    npt.assert_raises(ValueError, fetch_gltf, name=["duck"])
    npt.assert_raises(ValueError, fetch_gltf, name=["Duck"], mode="GLTF")

    fetch_gltf()
    list_gltf = os.listdir(folder)
    default_list = ["BoxTextured", "Duck", "CesiumMilkTruck", "CesiumMan"]
    results = [model in list_gltf for model in default_list]
    npt.assert_equal(results, [True, True, True, True])

    items = os.listdir(boxtex)
    npt.assert_array_equal(len(items), 3)

    filenames, path = fetch_gltf(name="Box", mode="glTF-Binary")
    npt.assert_equal(len(filenames), 1)
    npt.assert_equal(os.listdir(path), filenames)

    gltf = pjoin(boxtex, "BoxTextured.gltf")
    with open(gltf, "r") as f:
        gltf = json.loads(f.read())
    validate_gltf = gltf.get("asset")
    npt.assert_equal(validate_gltf["version"], str(2.0))


def test_list_gltf_sample_models():
    fetch_names = list_gltf_sample_models()
    default_list = ["BoxTextured", "Duck", "CesiumMilkTruck", "CesiumMan"]
    result = [model in fetch_names for model in default_list]
    npt.assert_equal(result, [True, True, True, True])


def test_read_viz_gltf():
    gltf_dir = pjoin(fury_home, "glTF")
    filenames, path = fetch_gltf(name="Box", mode="glTF-Binary")
    filename = read_viz_gltf("Box", mode="glTF-Binary")
    npt.assert_equal(filename, pjoin(path, filenames[0]))

    npt.assert_raises(ValueError, read_viz_gltf, "FURY", mode="glTF")

    box_gltf = pjoin(gltf_dir, "Box")
    for path in os.listdir(box_gltf):
        mode = pjoin(box_gltf, path)
        for file in os.listdir(mode):
            os.remove(pjoin(mode, file))
        os.rmdir(mode)
    npt.assert_raises(ValueError, read_viz_gltf, "Box")

    filenames, path = fetch_gltf(name="Box")
    out_path = read_viz_gltf("Box").split(os.sep)
    mode = out_path[-2:][0]
    npt.assert_equal(mode, "glTF")


def _serve_theme(server, name, content):
    """Write a theme file into the served folder and return its checksum."""
    data = content.encode("utf-8")
    (server.folder / f"{name}.json").write_bytes(data)
    return sha256(data).hexdigest()


def test_ui_theme_registry_points_at_fury_data():
    """The listed UI themes have SHA-256 checksums and a fury-data address."""
    assert UI_THEMES_DATA_URL.startswith(
        "https://raw.githubusercontent.com/fury-gl/fury-data/"
    )
    assert UI_THEMES_DATA_URL.endswith("/ui_themes/")
    assert UI_THEME_SHA256
    for name, checksum in UI_THEME_SHA256.items():
        assert name.isidentifier() and name.islower()
        assert len(checksum) == 64
        int(checksum, 16)


def test_fetch_ui_themes_downloads_once_and_caches(local_http_server, tmp_path):
    """A theme is downloaded once; a later fetch reuses the verified file."""
    checksum = _serve_theme(local_http_server, "night", '{"name": "night"}')
    cache = tmp_path / "cache"

    files, folder = fetch_ui_themes(
        "night",
        base_url=local_http_server.url,
        folder=str(cache),
        sha256={"night": checksum},
    )

    assert folder == str(cache)
    assert files == {"night.json": (local_http_server.url + "night.json", checksum)}
    assert (cache / "night.json").read_text() == '{"name": "night"}'
    assert local_http_server.requested_paths == ["/night.json"]

    (local_http_server.folder / "night.json").unlink()
    fetch_ui_themes(
        ["night"],
        base_url=local_http_server.url.rstrip("/"),
        folder=str(cache),
        sha256={"night": checksum},
    )
    assert local_http_server.requested_paths == ["/night.json"]


def test_fetch_ui_themes_redownloads_a_changed_cached_file(local_http_server, tmp_path):
    """A cached file that no longer matches its checksum is downloaded again."""
    checksum = _serve_theme(local_http_server, "night", '{"name": "night"}')
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "night.json").write_text('{"name": "edited"}')

    fetch_ui_themes(
        "night",
        base_url=local_http_server.url,
        folder=str(cache),
        sha256={"night": checksum},
    )

    assert (cache / "night.json").read_text() == '{"name": "night"}'
    assert local_http_server.requested_paths == ["/night.json"]


def test_fetch_ui_themes_rejects_a_checksum_mismatch(local_http_server, tmp_path):
    """A downloaded file that does not match its checksum raises FetcherError."""
    _serve_theme(local_http_server, "night", '{"name": "night"}')
    cache = tmp_path / "cache"

    with pytest.raises(FetcherError, match="expected sha"):
        fetch_ui_themes(
            "night",
            base_url=local_http_server.url,
            folder=str(cache),
            sha256={"night": "0" * 64},
        )
    assert cache.is_dir()
    assert not (cache / "night.json").exists()


def test_fetch_ui_themes_lets_local_errors_through(local_http_server, tmp_path):
    """A local file system error is not reported as a network failure."""
    checksum = _serve_theme(local_http_server, "night", '{"name": "night"}')
    not_a_folder = tmp_path / "cache"
    not_a_folder.write_text("a file, not a folder")

    with pytest.raises(NotADirectoryError):
        fetch_ui_themes(
            "night",
            base_url=local_http_server.url,
            folder=str(not_a_folder),
            sha256={"night": checksum},
        )


def test_fetch_ui_themes_reports_a_missing_file(local_http_server, tmp_path):
    """A file missing on the server raises FetcherError naming the theme."""
    with pytest.raises(FetcherError, match="Could not download UI themes night"):
        fetch_ui_themes(
            "night",
            base_url=local_http_server.url,
            folder=str(tmp_path / "cache"),
            sha256={"night": "0" * 64},
        )
    assert local_http_server.requested_paths == ["/night.json"]
    assert not (tmp_path / "cache" / "night.json").exists()


def test_fetch_ui_themes_reports_an_unreachable_server(tmp_path):
    """A server that refuses connections raises FetcherError."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with pytest.raises(FetcherError, match="Check the network connection"):
        fetch_ui_themes(
            "night",
            base_url=f"http://127.0.0.1:{port}/",
            folder=str(tmp_path / "cache"),
            sha256={"night": "0" * 64},
        )


@pytest.mark.parametrize("name", ["", "../x", "C:foo"])
def test_fetch_ui_themes_rejects_names_that_are_not_plain(name, tmp_path):
    """Empty names, folders, drives and colons raise before any download."""
    with pytest.raises(ValueError, match="Invalid UI theme name"):
        fetch_ui_themes(name, folder=str(tmp_path), sha256={name: "0" * 64})
    assert not os.listdir(tmp_path)


def test_fetch_ui_themes_rejects_unknown_names(tmp_path):
    """Names without a checksum raise a ValueError listing the known themes."""
    with pytest.raises(ValueError, match="Unknown UI themes: missing.*slate"):
        fetch_ui_themes("missing", folder=str(tmp_path))
    with pytest.raises(ValueError, match="Downloadable themes: night"):
        fetch_ui_themes(["missing"], folder=str(tmp_path), sha256={"night": "0"})
    assert not os.listdir(tmp_path)
