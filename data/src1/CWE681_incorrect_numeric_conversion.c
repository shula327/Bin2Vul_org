// 不安全：不正确数值转换导致丢失精度
float convert_unsafe(double value) {
    return (float)value;  // 不正确数值转化，可能丢失精度
} 