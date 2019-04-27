sudo -u postgres psql <<PGSCRIPT
CREATE database $1;
GRANT ALL PRIVILEGES on database holo to holocleanUser ;
PGSCRIPT

echo "PG database and user has been created."
