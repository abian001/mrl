quick_tests:
	python3 -m pytest . -m 'quick' 

slow_tests:
	python3 -m pytest . -m 'slow'

performance_tests:
	python3 -m pytest . -m 'performance'

manual_tests:
	python3 -m pytest . -m 'manual' -s

gui_tests:
	python3 -m pytest . -m 'gui'

debug_tests:
	python3 -m pytest . -m 'debug' -s -vv

PYTHON_MODULES := source/mrl

pylint:
	pylint $(PYTHON_MODULES)

pyright:
	pyright $(PYTHON_MODULES)

mypy:
	mypy $(PYTHON_MODULES)

doc: doc/source/*.py doc/source/*.rst
	sphinx-build -M html doc/source doc/build

format_doc: doc/source/*.rst
	rstfmt doc/source/*.rst

profile_debug_tests:
	python3 -m cProfile -s cumtime -o profile_record.out -m pytest . -m 'debug'
	gprof2dot -f pstats profile_record.out | dot -Tpng -o profile_record.png
	rm profile_record.out

build_dev_image:
	docker compose build mrl_dev

build_prod_image:
	docker compose build mrl_prod

run_dev_container:
	docker compose run --rm mrl_dev

run_prod_container:
	docker compose run --rm mrl_prod
