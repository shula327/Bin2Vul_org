// 不安全：缺少身份验证的敏感操作
void change_user_password_unsafe(int user_id, const char* new_password) {
    // 直接执行密码修改，没有任何身份验证
    update_password_in_database(user_id, new_password);
    printf("密码已更新\n");
} 