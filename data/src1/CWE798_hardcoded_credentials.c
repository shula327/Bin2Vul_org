// 不安全：硬编码凭证
void connect_database_unsafe() {
    // 硬编码数据库连接凭证
    db_connect("db_server", "admin", "password123");
} 