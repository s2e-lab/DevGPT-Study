public class FileFrameController {
  @AuraEnabled(cacheable=true)
  public static String getFileURL(Id recordId) {
      ContentVersion version = [
          SELECT Id, ContentDocumentId 
          FROM ContentVersion 
          WHERE ContentDocument.LatestPublishedVersion.LinkedEntityId = :recordId 
          ORDER BY CreatedDate DESC 
          LIMIT 1
      ];
      String fileURL = '/sfc/servlet.shepherd/version/renditionDownload?rendition=SVGZ&versionId=' + version.Id + '&operationContext=CHATTER&contentId=' + version.ContentDocumentId + '&page=0';
      return fileURL;
  }
}
