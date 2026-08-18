.PHONY: check

check:
	python3 scripts/validate_config.py
	python3 -m unittest discover -s tests -v
	PYTHONPYCACHEPREFIX=/tmp/oellm-reasoning-pycache python3 -m compileall -q scripts tests
