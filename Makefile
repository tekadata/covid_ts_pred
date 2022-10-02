# directive_name:
# <tab>some command with all its arguments
# <tab>@this command will not print out before being executed
# <tab>-the command after this one will run no matter what
# <tab>-@the markers can be combined

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* covid_ts_pred/*.py

black:
	@black scripts/* covid_ts_pred/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr covid_ts_pred-*.dist-info
	@rm -fr covid_ts_pred.egg-info
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc


re_install_package:
	@pip uninstall -y covid_ts_pred || :
	@pip install -e .

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
run_preprocess:
	python -c 'from covid_ts_pred.c_eng.main import preprocess; preprocess(); preprocess(source_type="val")'

run_train:
	python -c 'from covid_ts_pred.c_eng.main import train; train()'

run_pred:
	python -c 'from covid_ts_pred.c_eng.main import pred; pred()'

run_evaluate:
	python -c 'from covid_ts_pred.c_eng.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

streamlit:
	-@streamlit run app.py

heroku_login:
	-@heroku login

heroku_upload_public_key:
	-@heroku keys:add ~/.ssh/id_ed25519.pub

heroku_create_app:
	-@heroku create --ssh-git ${APP_NAME}

deploy_heroku:
	-@git push heroku master
	-@heroku ps:scale web=1

##################### DEBUGGING HELPERS ####################
fbold=$(shell echo "\033[1m")
fnormal=$(shell echo "\033[0m")
ccgreen=$(shell echo "\033[0;32m")
ccblue=$(shell echo "\033[0;34m")
ccreset=$(shell echo "\033[0;39m")

show_env:
	@echo "\nEnvironment variables used by the \`covid_ts_pred\` package loaded by \`direnv\` from your \`.env\` located at:"
	@echo ${DIRENV_DIR}

	@echo "\n$(ccgreen)local storage:$(ccreset)"
	@env | grep -E "LOCAL_DATA_PATH|LOCAL_REGISTRY_PATH" || :
	@echo "\n$(ccgreen)dataset:$(ccreset)"
	@env | grep -E "DATASET_SIZE|VALIDATION_DATASET_SIZE|CHUNK_SIZE" || :
	@echo "\n$(ccgreen)package behavior:$(ccreset)"
	@env | grep -E "DATA_SOURCE|MODEL_TARGET" || :

list:
	@echo "\nHelp for the \`covid_ts_pred\` package \`Makefile\`"

	@echo "\n$(ccgreen)$(fbold)PACKAGE$(ccreset)"

	@echo "\n$(ccgreen)$(fbold)TESTS$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)student rules:$(ccreset)"
	@echo "\n        $(fbold)reinstall_package$(ccreset)"
	@echo "            Install the version of the package corresponding to the challenge."
