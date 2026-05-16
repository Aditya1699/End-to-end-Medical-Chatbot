from app import app


if __name__ == "__main__":
    from waitress import serve

    config = app.config_obj
    serve(app, host=config.host, port=config.port)
