# ============================================================
#  db_config.py  —  DATABASE CREDENTIALS
# ============================================================
#  This is the ONLY file you edit to change the database
#  connection. Every notebook reads its credentials from here.
# ============================================================

DB_HOST     = "database-1.cx86gkkcmec6.ap-south-1.rds.amazonaws.com"
DB_PORT     = "5432"
DB_NAME     = "smartsentry_aml"          # or "postgres"
DB_USER     = "postgres"
DB_PASSWORD = "LIZP4vOH4WrZ5N6Y"             # <-- put the real password here

# AWS RDS requires SSL. Leave as "require" for RDS.
DB_SSLMODE  = "require"

# Schema that holds the pipeline tables.
DB_SCHEMA   = "public"
