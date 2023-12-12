import org.objectweb.asm.*;

public class ExecutorClassGenerator {

    // ... [This includes all the ASM bytecode generation code I provided earlier]

    public static DynamicExecutor createExecutorFor(Class<?> targetClass) throws Exception {
        byte[] classData = generateExecutorFor(targetClass);
        Class<?> generatedClass = new ClassLoader() {
            public Class<?> defineClass(String name, byte[] b) {
                return defineClass(name, b, 0, b.length);
            }
        }.defineClass("GeneratedExecutor", classData);
        
        return (DynamicExecutor) generatedClass.getConstructor(targetClass).newInstance(targetClass.newInstance());
    }
}
