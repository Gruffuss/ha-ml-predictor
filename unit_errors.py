==================================== ERRORS ====================================
________ ERROR at setup of TestDatabaseManager.test_get_session_success ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_______ ERROR at setup of TestDatabaseManager.test_execute_query_success _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
________ ERROR at setup of TestDatabaseManager.test_execute_query_error ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_______ ERROR at setup of TestDatabaseManager.test_health_check_healthy ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
______ ERROR at setup of TestDatabaseManager.test_health_check_unhealthy _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_ ERROR at setup of TestDatabaseManager.test_health_check_timescaledb_available _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_ ERROR at setup of TestDatabaseManager.test_health_check_timescaledb_unavailable _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_ ERROR at setup of TestDatabaseManager.test_health_check_timescaledb_version_parsing _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_ ERROR at setup of TestDatabaseManager.test_health_check_timescaledb_version_parsing_error _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_ ERROR at setup of TestDatabaseManagerEdgeCases.test_health_check_with_previous_errors _
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
___________ ERROR at setup of TestSensorEvent.test_get_recent_events ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_ ERROR at setup of TestSensorEvent.test_get_recent_events_with_sensor_filter __
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
___________ ERROR at setup of TestSensorEvent.test_get_state_changes ___________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_______ ERROR at setup of TestSensorEvent.test_get_transition_sequences ________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
____________ ERROR at setup of TestRoomState.test_get_current_state ____________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
_______ ERROR at setup of TestRoomState.test_get_current_state_not_found _______
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
__________ ERROR at setup of TestRoomState.test_get_occupancy_history __________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
    res = await gen_obj.__anext__()
tests/conftest.py:258: in test_db_engine
    await conn.run_sync(Base.metadata.create_all)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/ext/asyncio/engine.py:887: in run_sync
    return await greenlet_spawn(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/util/_concurrency_py3k.py:203: in greenlet_spawn
    result = context.switch(value)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/schema.py:5924: in create_all
    bind._run_ddl_visitor(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:2459: in _run_ddl_visitor
    ).traverse_single(element)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:984: in visit_metadata
    self.traverse_single(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:661: in traverse_single
    return meth(obj, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:1022: in visit_table
    )._invoke_with(self.connection)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:321: in _invoke_with
    return bind.execute(self)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1419: in execute
    return meth(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:187: in _execute_on_connection
    return connection._execute_ddl(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/engine/base.py:1527: in _execute_ddl
    compiled = ddl.compile(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/elements.py:311: in compile
    return self._compiler(dialect, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/ddl.py:76: in _compiler
    return dialect.ddl_compiler(dialect, self, **kw)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:886: in __init__
    self.string = self.process(self.statement, **compile_kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6779: in visit_create_table
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: (in table 'sensor_events', column 'id'): SQLite does not support autoincrement for composite primary keys
________ ERROR at setup of TestPrediction.test_get_pending_validations _________
[gw0] linux -- Python 3.12.11 /opt/hostedtoolcache/Python/3.12.11/x64/bin/python
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6769: in visit_create_table
    processed = self.process(
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:932: in process
    return obj._compiler_dispatch(self, **kwargs)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/visitors.py:138: in _compiler_dispatch
    return meth(self, **kw)  # type: ignore  # noqa: E501
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/sql/compiler.py:6800: in visit_create_column
    text = self.get_column_specification(column, first_pk=first_pk)
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/sqlalchemy/dialects/sqlite/base.py:1702: in get_column_specification
    raise exc.CompileError(
E   sqlalchemy.exc.CompileError: SQLite does not support autoincrement for composite primary keys
The above exception was the direct cause of the following exception:
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:347: in _asyncgen_fixture_wrapper
    result = event_loop.run_until_complete(setup())
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
/opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages/pytest_asyncio/plugin.py:329: in setup
Error: Process completed with exit code 1.