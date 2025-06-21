// 不安全：不充分的授权检查
int view_document_unsafe(int user_id, int document_id) {
    // 检查用户是否已登录，但没有检查用户是否有权限查看此文档
    if (is_user_logged_in(user_id)) {
        return get_document_content(document_id);
    }
    return -1; // 未认证
} 