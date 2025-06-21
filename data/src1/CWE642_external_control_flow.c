// 不安全：使用用户可控数据进行安全决策
void process_request_unsafe(Request* req) {
    // 从客户端获取角色信息，而不是从会话获取
    char* role = get_header(req, "X-User-Role");
    
    // 基于不可信数据授予权限
    if (role && strcmp(role, "admin") == 0) {
        // 授予管理员权限
        grant_admin_access(req);
    } else {
        // 授予普通用户权限
        grant_user_access(req);
    }
} 