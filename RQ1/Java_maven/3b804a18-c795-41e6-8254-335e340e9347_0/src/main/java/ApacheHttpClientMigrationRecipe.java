import org.openrewrite.*;
import org.openrewrite.java.JavaIsoVisitor;
import org.openrewrite.java.JavaVisitor;
import org.openrewrite.java.MethodMatcher;
import org.openrewrite.java.tree.J;
import org.openrewrite.java.tree.JavaType;

public class ApacheHttpClientMigrationRecipe extends Recipe {
    private static final MethodMatcher CLIENT_EXECUTION_METHOD =
            new MethodMatcher("org.apache.http.client.HttpClient execute*(..)");

    @Override
    public String getDisplayName() {
        return "Apache HttpClient Migration";
    }

    @Override
    public String getDescription() {
        return "Migrates Apache HttpClient 4 usages to HttpClient 5";
    }

    @Override
    protected TreeVisitor<?, ExecutionContext> getSingleSourceApplicableTest() {
        return new JavaVisitor<ExecutionContext>() {
            @Override
            public J visitMethodInvocation(J.MethodInvocation method, ExecutionContext ctx) {
                if (CLIENT_EXECUTION_METHOD.matches(method)) {
                    return method.withTemplate(
                            JavaTemplate.builder(this::getCursor, "httpClient.execute(request, ResponseHandlerRegistry.<T>ofDefault())")
                                    .imports("org.apache.hc.client5.http.classic.methods.HttpGet",
                                            "org.apache.hc.core5.http.client.ResponseHandlerRegistry")
                                    .javaParser(getCursor().getTree().getJavaParser())
                                    .build()
                    ).withArguments(method.getArguments().get(0));
                }
                return super.visitMethodInvocation(method, ctx);
            }
        };
    }

    @Override
    protected TreeVisitor<?, ExecutionContext> getVisitor() {
        return new JavaIsoVisitor<ExecutionContext>() {
            @Override
            public J.MethodInvocation visitMethodInvocation(J.MethodInvocation method, ExecutionContext ctx) {
                // Replace deprecated method calls
                if (method.getSimpleName().equals("create")) {
                    JavaType.Class type = JavaType.Class.build("org.apache.http.impl.client.HttpClients");
                    maybeRemoveImport("org.apache.http.impl.client.HttpClients");
                    return method.withSelect(J.Identifier.build(type.getClassName()))
                            .withName(J.Identifier.build("custom"));
                }
                return super.visitMethodInvocation(method, ctx);
            }
        };
    }
}
