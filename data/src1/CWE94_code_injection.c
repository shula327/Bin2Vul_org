// 不安全：动态代码执行
void execute_script_unsafe(const char* user_script) {
    // 直接执行用户提供的脚本代码
    eval(user_script); // 假设eval函数可执行代码字符串
    // 或者：system(("python -c \"" + user_script + "\"").c_str());
} 