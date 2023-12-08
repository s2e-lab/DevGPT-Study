import org.objectweb.asm.*;

public class ExecutorClassGenerator {

    public static byte[] generateExecutorFor(Class<?> targetClass) {
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        FieldVisitor fv;
        MethodVisitor mv;

        String className = "GeneratedExecutor";
        String classDescriptor = "L" + className + ";";
        String targetClassDescriptor = "L" + targetClass.getName().replace(".", "/") + ";";

        cw.visit(Opcodes.V1_8, Opcodes.ACC_PUBLIC + Opcodes.ACC_SUPER, className, null, "java/lang/Object", new String[]{"DynamicExecutor"});

        // Field for the subscriber instance
        fv = cw.visitField(Opcodes.ACC_PRIVATE, "subscriber", targetClassDescriptor, null, null);
        fv.visitEnd();

        // Constructor
        mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "(" + targetClassDescriptor + ")V", null, null);
        mv.visitCode();
        mv.visitVarInsn(Opcodes.ALOAD, 0); // this
        mv.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitVarInsn(Opcodes.ALOAD, 0); // this
        mv.visitVarInsn(Opcodes.ALOAD, 1); // 1st argument (subscriber)
        mv.visitFieldInsn(Opcodes.PUTFIELD, className, "subscriber", targetClassDescriptor);
        mv.visitInsn(Opcodes.RETURN);
        mv.visitMaxs(2, 2);
        mv.visitEnd();

        // Implement the 'execute' method
        mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "execute", "(LEvent;[Ljava/lang/Object;)V", null, null);
        mv.visitCode();

        for (java.lang.reflect.Method method : targetClass.getDeclaredMethods()) {
            if (method.isAnnotationPresent(Observe.class)) {
                Class<?>[] params = method.getParameterTypes();
                Class<?> eventType = params[0];

                Label skipInvocation = new Label();

                mv.visitVarInsn(Opcodes.ALOAD, 1);
                mv.visitTypeInsn(Opcodes.INSTANCEOF, eventType.getName().replace(".", "/"));
                mv.visitJumpInsn(Opcodes.IFEQ, skipInvocation);

                mv.visitVarInsn(Opcodes.ALOAD, 0); // this
                mv.visitFieldInsn(Opcodes.GETFIELD, className, "subscriber", targetClassDescriptor);
                mv.visitVarInsn(Opcodes.ALOAD, 1);
                mv.visitTypeInsn(Opcodes.CHECKCAST, eventType.getName().replace(".", "/"));

                // Load additional parameters
                for (int i = 1; i < params.length; i++) {
                    Class<?> paramType = params[i];
                    Label endOfLoop = new Label();
                    Label paramFound = new Label();

                    mv.visitInsn(Opcodes.ICONST_0); // Counter set to 0
                    mv.visitVarInsn(Opcodes.ISTORE, 3);

                    mv.visitLabel(endOfLoop);
                    mv.visitVarInsn(Opcodes.ILOAD, 3);
                    mv.visitVarInsn(Opcodes.ALOAD, 2); // additionalArgs array
                    mv.visitInsn(Opcodes.ARRAYLENGTH);
                    mv.visitJumpInsn(Opcodes.IF_ICMPGE, paramFound); 

                    mv.visitVarInsn(Opcodes.ALOAD, 2);
                    mv.visitVarInsn(Opcodes.ILOAD, 3);
                    mv.visitInsn(Opcodes.AALOAD);

                    mv.visitInsn(Opcodes.DUP);
                    mv.visitTypeInsn(Opcodes.INSTANCEOF, paramType.getName().replace(".", "/"));
                    mv.visitJumpInsn(Opcodes.IFNE, paramFound); 

                    mv.visitIincInsn(3, 1); 
                    mv.visitJumpInsn(Opcodes.GOTO, endOfLoop);

                    mv.visitLabel(paramFound);
                    mv.visitTypeInsn(Opcodes.CHECKCAST, paramType.getName().replace(".", "/"));
                }

                mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, targetClass.getName().replace(".", "/"), method.getName(), Type.getMethodDescriptor(method), false);
                mv.visitLabel(skipInvocation);
            }
        }

        mv.visitInsn(Opcodes.RETURN);
        mv.visitMaxs(-1, -1); 
        mv.visitEnd();

        cw.visitEnd();
        return cw.toByteArray();
    }
}
