"""Study Hub management commands (run from the repository folder):

    python -m hub.manage init-db
    python -m hub.manage create-pi "Dr Name" pi@example.org
    python -m hub.manage run           (development server on port 5000)

STUDYHUB_DATA sets where study.db and answer files live (default:
./hub_data next to where you run the command)."""

import getpass
import sys

from werkzeug.security import generate_password_hash

from . import create_app
from .auth import create_user
from .db import connect


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    command = argv[0]
    app = create_app()

    if command == "init-db":
        # create_app already ran the migrations.
        print("Database ready at {}".format(app.config["DATABASE"]))
        return 0

    if command == "create-pi":
        if len(argv) != 3:
            print('Usage: python -m hub.manage create-pi "Name" email@example.org')
            return 1
        name, email = argv[1], argv[2]
        password = getpass.getpass("Choose the PI password (8+ characters): ")
        if len(password) < 8:
            print("Password too short.")
            return 1
        db = connect(app.config["DATABASE"])
        try:
            if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                print("There is already an account with that email.")
                return 1
            user_id, _token = create_user(db, name, email, "pi", invited=False)
            db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(password), user_id),
            )
            db.commit()
        finally:
            db.close()
        print("PI account created for {} <{}>.".format(name, email))
        return 0

    if command == "run":
        app.run(host="127.0.0.1", port=5000, debug=False)
        return 0

    print("Unknown command: {}".format(command))
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
