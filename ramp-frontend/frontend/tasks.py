from invoke import task


@task
def launch(c):
    from application import create_app
    app = create_app()
    app.run(debug=False, port=8080, use_reloader=False,
            host='127.0.0.1', processes=1000, threaded=False)
