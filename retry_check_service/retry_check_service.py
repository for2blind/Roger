from quart import Quart, request, jsonify
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
from collections import Counter
import uvloop

uvloop.install()

app = Quart(__name__)

uid_requests_count = Counter()


@app.route("/retry_check", methods=["POST"])
async def retry_check():
    try:
        request_data = await request.get_json()

        if not request_data:
            return jsonify({"error": "No JSON data received"}), 400
        print(request_data)
        uid = request_data.get("uid")
        if not uid:
            return jsonify({"error": "No UID provided"}), 400
        count = uid_requests_count[uid]
        if count >= 3:
            return jsonify({"retry": False}), 200

        stage = float(request_data.get("stage", 0))
        pipeline_length = float(request_data.get("pipeline_length", 0))

        if stage < 0 or pipeline_length <= 0:
            return jsonify({"error": "Invalid stage or pipeline_length value"}), 400

        if stage < pipeline_length / 2.71828:  # e ≈ 2.71828
            uid_requests_count[uid] += 1
            return jsonify({"retry": True}), 200
        else:
            return jsonify({"retry": False}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:80"]
    config.use_reloader = True
    asyncio.run(serve(app, config))
