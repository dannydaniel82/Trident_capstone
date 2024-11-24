// input file 커스텀 - 파일명 붙이기
  const fileTarget = $('.form__input--file_wrap input');

  fileTarget.on('change', function () { 
    var files = $(this)[0].files;
    var fileArr = [];
    for (var i = 0; i < files.length; i++) {
      fileArr.push(files[i].name);
    }

  // 파일명 노출방법2: 배열 값들을 줄바꿈하여 표시
  var fileList = fileArr.join('<br>');
  $(this).siblings('.form__span--file').html(fileList);

});