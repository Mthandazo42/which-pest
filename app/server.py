from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *

import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

#assign the path variable to be the parent directory of the project
path = Path(__file__).parent
#url link which contains the exported model.
model_url = 'https://drive.google.com/uc?export=download&id=1AC_ZbN5PIaAWG_w4vuyN68qtI3c7UNOr'
#our model export name
model_name = "export.pkl"
#the classes present in our model
classes = {'wireworm':'Patchy seedling emergence, wilting and tillering of seedlings, and lodging of older plants are signs of wireworm injury, although these symptoms may also be associated with other soil insects', 'white_grubs':'Root damage by white grubs is evidenced first by wilting seedlings and later by poor stands and patches of tilted, curved, or lodged plants that show uneven growth', 'spider_mites':'The presence of small, faint, yellow blotches on the lower leaves is an indication of spider mite injury, which is inflicted through piercing and sucking of the foliar tissue','maize_billbug':'Leaves show white specks, which grow together under severe infestation. The specks are an indication of feeding by billbugs in the whorl, where they scratch small, irregular sections of the epidermis without puncturing it ', 'lesser_cornstalk_borer':'Early damage to the seedling produces a  series of holes that become visible as leaves unfold','flea_beetle':'On recently emerged seedlings, the main symptom of attack is white, thin, elongated lesions (injured areas) along the upper leaf surface, which are the result of scratching between the leaf veins by the beetles','fall_armyworm':'Extensive leaf damage, which becomes quite noticeable as the leaves unfold, is caused by the small, dark-green worms. Upon hatching they begin to feed by scraping the leaf epidermis and later migrate to the whorl, where they feed voraciously','cutworm':'Young cutworms (and some other species) cut maize seedlings at or a little below ground level, make small holes along the initial leaves, or remove sections from the leaf margins','corn_leaf_aphid':'Diseased plants may become stunted, show a conspicuous yellowish mottling, and turn reddish as they mature. Young plants that have been infected seldom produce ears','armyworm':'Starting at the margins and moving inward, larvae may eat entire leaves, leaving only the midribs. Under severe infestation, the entire young plant may be consumed.','grasshoppers':'These insects attack maize from the midwhorl stage to maturity and consume every part of the plant'}

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

async def setup_learner():
    await download_file(model_url, path / model_name)
    try:
        learner = load_learner(path, model_name)
        return learner
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message= 'Please upgrade the model'
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learner = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learner.predict(img)
    res = classes[str(prediction[0])]
    message = f'Pest: {prediction[0]}, Nature of damage: {res}'
    result_base = path/'view'/'base_result.html'
    results = path/'view'/'result.html'
    result_main = str(result_base.open().read() + str(message) + results.open().read())
    return HTMLResponse(result_main)
    # return JSONResponse({'result': str(prediction[0])})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
