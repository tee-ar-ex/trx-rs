.PHONY: docs docs-serve docs-clean docs-linkcheck

docs:
	cargo doc --no-deps --target-dir target/rustdoc
	.venv/bin/sphinx-build docs/ docs/_build/html
	mkdir -p docs/_build/html/api/rust
	cp -r target/rustdoc/doc/* docs/_build/html/api/rust/

docs-serve: docs
	python3 -m http.server 8000 -d docs/_build/html

docs-clean:
	rm -rf docs/_build target/rustdoc

docs-linkcheck:
	.venv/bin/sphinx-build -b linkcheck docs/ docs/_build/linkcheck
