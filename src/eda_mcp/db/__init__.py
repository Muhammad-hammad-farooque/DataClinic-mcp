"""Database access: connections, introspection, push-down profiling, the SQL guard.

Everything here needs the ``sql`` extra (SQLAlchemy and sqlglot); a missing
driver surfaces as ``DEPENDENCY_MISSING`` naming the extra to install.

See spec section 6.
"""
