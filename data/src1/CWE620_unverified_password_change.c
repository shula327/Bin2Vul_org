// 不安全：密码更改没有验证旧密码
void change_password_unsafe(int user_id, const char* new_password) {
    // 直接更改密码，没有要求提供当前密码进行验证
    set_user_password(user_id, new_password);
    printf("密码已更新\n");
} 