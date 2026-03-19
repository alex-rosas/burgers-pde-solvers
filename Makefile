.PHONY: all analysis test app clean

all: analysis test

analysis:
	python analysis/convergence.py
	python analysis/performance.py
	python analysis/shock.py
	python analysis/cfl.py
	python analysis/formulation.py

test:
	pytest tests/ -v

app:
	streamlit run app.py

clean:
	rm -f figures/*.png results/*.csv
