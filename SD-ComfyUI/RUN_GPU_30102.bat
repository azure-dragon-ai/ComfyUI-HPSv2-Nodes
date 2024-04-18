set HF_ENDPOINT=https://hf-mirror.com
set HPS_ROOT=\\NAS65A682\SD-Share\models\Score\HPSv2Models
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --cuda-device=1 --highvram --force-fp16 --listen=[2408:8207:60ad:6cb0::6aa] --port=30102 --output-directory=\\NAS65A682\Web\images\test
pause