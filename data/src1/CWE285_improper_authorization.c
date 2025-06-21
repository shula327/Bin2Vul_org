// 不安全：不恰当的访问控制
int delete_user_unsafe(int user_id, int target_id) {
    // 没有验证操作用户是否有删除其他用户的权限
    // 例如，普通用户可能删除管理员账户
    return database_delete_user(target_id);
} 