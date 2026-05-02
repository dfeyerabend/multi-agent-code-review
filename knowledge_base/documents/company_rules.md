# example_company_code_standards — Internal Python Coding Rules

These rules apply to all Python code written at example_company. They exist alongside
PEP 8 and the Google Python Style Guide, which remain the baseline standard.
Violations of these rules must be flagged in every code review.

---

### 1.1 Database Function Naming Convention

All functions that interact with a database must be prefixed with db_fetch_, db_save_,
or db_delete_ depending on their operation. This makes data access points immediately
visible during code review and grep-searchable across the codebase.

Allowed prefixes:
- db_fetch_ — read operations (SELECT)
- db_save_  — insert or update operations (INSERT, UPDATE)
- db_delete_ — delete operations (DELETE)

BAD — violates example_company naming standard:

    def get_user(user_id):
        return db.query(f"SELECT * FROM users WHERE id = {user_id}")

    def user_save(data):
        db.execute("INSERT INTO users VALUES (?)", data)

GOOD — compliant with example_company naming standard:

    def db_fetch_user(user_id):
        return db.query("SELECT * FROM users WHERE id = ?", (user_id,))

    def db_save_user(data):
        db.execute("INSERT INTO users VALUES (?)", (data,))


---

### 1.2 Required REASON Comment for Non-Trivial Functions

Every non-trivial function must include a # REASON: inline comment on or directly
below the def line. The comment must explain WHY the function exists, not what
it does (that belongs in the docstring). This rule exists to prevent orphaned utility
functions whose original purpose has been forgotten.

BAD — missing REASON comment:

    def retry_request(url, retries=3):
        """Retries an HTTP request up to n times."""
        ...

GOOD — includes REASON comment:

    def retry_request(url, retries=3):  # REASON: external payment API times out under peak load
        """Retries an HTTP request up to n times."""
        ...


---

### 1.3 Custom Exception Hierarchy — no bare Exception raises

Never raise Python built-in exceptions (Exception, ValueError, RuntimeError) directly.
All raised exceptions must be instances of AppError or one of its registered subclasses.
This ensures that all errors are loggable, traceable, and handleable uniformly at the
application boundary.

Registered subclasses:
- AppError           — base class for all application errors
- AppValidationError — invalid input or schema mismatch
- AppConfigError     — missing or malformed configuration values
- AppNotFoundError   — resource not found

BAD — raises built-in exception:

    def load_config(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")

GOOD — raises company exception class:

    def load_config(path):
        if not os.path.exists(path):
            raise AppConfigError(f"Config not found: {path}")


---

### 1.4 Config and Secret Access — no direct os.environ

Never access environment variables directly via os.environ["KEY"] or os.getenv("KEY").
All configuration and secret access must go through the project's Config.get("KEY")
wrapper. This centralizes error handling, provides default values, and ensures secrets
are logged and audited consistently.

BAD — direct environment access:

    import os
    api_key = os.environ["PAYMENT_API_KEY"]
    db_url = os.getenv("DATABASE_URL")

GOOD — access via Config wrapper:

    from config import Config
    api_key = Config.get("PAYMENT_API_KEY")
    db_url = Config.get("DATABASE_URL")
