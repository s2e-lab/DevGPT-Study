public class Main {
    public static void main(string [] args) {
        byte[] privateKeyBytes = Files.readAllBytes(Paths.get("path/to/jwtRS256-private.pk8"));
        PKCS8EncodedKeySpec privateKeySpec = new PKCS8EncodedKeySpec(privateKeyBytes);
        rsaPrivateKey = (RSAPrivateKey) keyFactory.generatePrivate(privateKeySpec);

        byte[] publicKeyBytes = Files.readAllBytes(Paths.get("path/to/jwtRS256-public.pem"));
        X509EncodedKeySpec publicKeySpec = new X509EncodedKeySpec(publicKeyBytes);
        rsaPublicKey = (RSAPublicKey) keyFactory.generatePublic(publicKeySpec);
   }
}
