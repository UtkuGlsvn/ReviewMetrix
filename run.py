import os

from reviewMetrix import create_app

app = create_app()

if __name__ == '__main__':
    # Debug modu varsayılan olarak kapalı; yerel geliştirmede FLASK_DEBUG=1 ile açılır.
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 4999))
    app.run(debug=debug, port=port)
