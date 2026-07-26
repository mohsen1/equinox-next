CREATE USER equinox_science WITH PASSWORD 'science-local-only';
CREATE USER equinox_ops WITH PASSWORD 'ops-local-only';
CREATE DATABASE equinox OWNER postgres;

\connect equinox

CREATE SCHEMA scientific AUTHORIZATION equinox_science;
CREATE SCHEMA operational AUTHORIZATION equinox_ops;
ALTER ROLE equinox_science IN DATABASE equinox SET search_path TO scientific, public;
ALTER ROLE equinox_ops IN DATABASE equinox SET search_path TO operational, public;
GRANT CONNECT ON DATABASE equinox TO equinox_science, equinox_ops;
